"""Run synchronized sounding and save timestamped CSI dataset chunks.

This is the experiment orchestration and persistence layer.  Waveform
generation/transmission stays in ``sounding_transmitter.py`` and receive DSP
stays in ``sounding_receiver.py``.

The transmitter continuously streams the prebuilt 9,600-sample period.  A
background RX worker continuously drains the receiver USRP while this main
thread performs ZC synchronization, LS estimation, metadata association, and
chunked dataset writes.  Python loop timing therefore does not define the
5 ms CSI interval; the UHD sample buffers and RX hardware timestamps do.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

if __package__:  # Package imports from the repository root.
    from .sounding_config import DEFAULT_SOUNDING_CONFIG, SoundingConfig
    from .sounding_receiver import (
        ContinuousRXCapture,
        RawCapture,
        ReceiverHardwareConfig,
        ReceiverProcessingConfig,
        SoundingCSIResult,
        initialize_receiver,
        process_received_samples,
    )
    from .sounding_transmitter import (
        DEFAULT_USRP_ADDRESSES,
        TransmitterHardwareConfig,
        build_sounding_period,
        build_transmitted_pilots,
        initialize_transmitter,
        transmit_sounding,
    )
else:  # Direct execution from inside USRP/.
    from sounding_config import DEFAULT_SOUNDING_CONFIG, SoundingConfig
    from sounding_receiver import (
        ContinuousRXCapture,
        RawCapture,
        ReceiverHardwareConfig,
        ReceiverProcessingConfig,
        SoundingCSIResult,
        initialize_receiver,
        process_received_samples,
    )
    from sounding_transmitter import (
        DEFAULT_USRP_ADDRESSES,
        TransmitterHardwareConfig,
        build_sounding_period,
        build_transmitted_pilots,
        initialize_transmitter,
        transmit_sounding,
    )


@dataclass(frozen=True, slots=True)
class ClockAnchor:
    """Approximate mapping between host UNIX and receiver USRP time."""

    usrp_time_s: float
    host_unix_time_s: float
    host_bracket_s: float


SnapshotMetadataProvider = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    Mapping[str, Any],
]


def map_usrp_to_host_unix(
    usrp_timestamps_s: NDArray[np.float64],
    clock_anchor: ClockAnchor,
) -> NDArray[np.float64]:
    """Map receiver-USRP time to approximate host UNIX time."""
    return (
        clock_anchor.host_unix_time_s
        + np.asarray(usrp_timestamps_s, dtype=np.float64)
        - clock_anchor.usrp_time_s
    )


def read_clock_anchor(usrp: Any) -> ClockAnchor:
    """Bracket a receiver USRP time read with host UNIX timestamps."""
    before_ns = time.time_ns()
    usrp_time_s = float(usrp.get_time_now().get_real_secs())
    after_ns = time.time_ns()
    return ClockAnchor(
        usrp_time_s=usrp_time_s,
        host_unix_time_s=(before_ns + after_ns) / 2e9,
        host_bracket_s=(after_ns - before_ns) / 1e9,
    )


def _import_uhd() -> Any:
    try:
        import uhd  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "UHD Python bindings are required to run the hardware experiment"
        ) from error
    return uhd


def _wait_for_pps_transition(usrp: Any, previous_pps_s: float) -> float:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        current = float(usrp.get_time_last_pps(0).get_real_secs())
        if current != previous_pps_s:
            return current
        time.sleep(0.01)
    raise TimeoutError("no external PPS transition was detected")


def synchronize_devices_on_common_pps(tx_usrp: Any, rx_usrp: Any) -> None:
    """Latch zero into all TX and RX device clocks on one physical PPS edge.

    Both calls are issued immediately after observing a PPS transition, so
    they target the same following edge from the common OctoClock output.
    """
    uhd = _import_uhd()
    previous_pps_s = float(tx_usrp.get_time_last_pps(0).get_real_secs())
    observed_pps_s = _wait_for_pps_transition(tx_usrp, previous_pps_s)

    zero_time = uhd.types.TimeSpec(0.0)
    tx_usrp.set_time_next_pps(zero_time)
    rx_usrp.set_time_next_pps(zero_time)
    _wait_for_pps_transition(tx_usrp, observed_pps_s)

    last_pps_values = [
        float(
            tx_usrp.get_time_last_pps(mboard).get_real_secs()
        )
        for mboard in range(tx_usrp.get_num_mboards())
    ]
    last_pps_values.append(
        float(rx_usrp.get_time_last_pps(0).get_real_secs())
    )
    if max(last_pps_values) - min(last_pps_values) > 1e-9:
        raise RuntimeError(
            "TX and RX USRPs did not latch the same PPS epoch: "
            f"{last_pps_values}"
        )
    if hasattr(tx_usrp, "get_time_synchronized"):
        if not tx_usrp.get_time_synchronized():
            raise RuntimeError("the TX MultiUSRP times are not synchronized")
    print("TX and RX UHD times synchronized on a common OctoClock PPS.")


def choose_experiment_start_time(
    tx_usrp: Any,
    rx_usrp: Any,
    minimum_lead_time_s: float,
) -> float:
    """Choose a future integer UHD time valid for both device objects."""
    latest_now = max(
        float(tx_usrp.get_time_now().get_real_secs()),
        float(rx_usrp.get_time_now().get_real_secs()),
    )
    return float(math.ceil(latest_now + minimum_lead_time_s) + 1)


def select_csi_result(
    result: SoundingCSIResult,
    selection: NDArray[np.bool_],
) -> SoundingCSIResult:
    """Select snapshots while preserving every aligned result field."""
    mask = np.asarray(selection, dtype=bool)
    if mask.shape != (result.num_snapshots,):
        raise ValueError(
            f"selection must have shape ({result.num_snapshots},), "
            f"got {mask.shape}"
        )
    return SoundingCSIResult(
        csi=result.csi[mask],
        csi_repetitions=result.csi_repetitions[mask],
        csi_repetitions_aligned=result.csi_repetitions_aligned[mask],
        pilot_received=result.pilot_received[mask],
        csi_timestamps_usrp_s=result.csi_timestamps_usrp_s[mask],
        pilot_timestamps_usrp_s=result.pilot_timestamps_usrp_s[mask],
        frame_start_sample_indices=result.frame_start_sample_indices[mask],
        frame_sequence_numbers=result.frame_sequence_numbers[mask],
        timing_error_samples=result.timing_error_samples[mask],
        zc_correlation_scores=result.zc_correlation_scores[mask],
        pilot_noise_power=result.pilot_noise_power[mask],
        pilot_snr_db=result.pilot_snr_db[mask],
        repetition_phase_corrections_rad=(
            result.repetition_phase_corrections_rad[mask]
        ),
        estimated_cfo_hz=result.estimated_cfo_hz,
    )


class CSIChunkWriter:
    """Persist one run as static metadata plus append-only NPZ chunks."""

    def __init__(
        self,
        output_directory: str | Path,
        config: SoundingConfig,
        tx_hardware: TransmitterHardwareConfig,
        rx_hardware: ReceiverHardwareConfig,
        processing: ReceiverProcessingConfig,
        clock_anchor: ClockAnchor,
        *,
        save_raw_iq: bool = False,
        rx_start_time_s: float | None = None,
        tx_start_time_s: float | None = None,
    ) -> None:
        created_at = datetime.now(timezone.utc)
        self.run_id = created_at.strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_directory = (
            Path(output_directory) / f"sounding_run_{self.run_id}"
        )
        self.run_directory.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.tx_hardware = tx_hardware
        self.rx_hardware = rx_hardware
        self.processing = processing
        self.clock_anchor = clock_anchor
        self.save_raw_iq = save_raw_iq
        self.chunk_index = 0
        self.total_snapshots = 0
        self.first_csi_timestamp_s: float | None = None
        self._manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "total_snapshots": 0,
            "chunks": [],
        }

        run_metadata = {
            "schema_version": 1,
            "run_id": self.run_id,
            "created_at_utc": created_at.isoformat(),
            "waveform_config": config.to_metadata(),
            "transmitter_hardware": asdict(tx_hardware),
            "receiver_hardware": asdict(rx_hardware),
            "receiver_processing": asdict(processing),
            "clock_anchor": asdict(clock_anchor),
            "experiment_timing": {
                "scheduled_rx_start_usrp_s": rx_start_time_s,
                "scheduled_tx_start_usrp_s": tx_start_time_s,
                "tx_minus_rx_start_s": (
                    None
                    if rx_start_time_s is None or tx_start_time_s is None
                    else tx_start_time_s - rx_start_time_s
                ),
            },
            "timestamp_convention": {
                "primary_field": "csi_timestamps_usrp_s",
                "primary_domain": "receiver_usrp_hardware_time",
                "mapped_field": "csi_timestamps_host_unix_s",
                "mapped_domain": "host_unix_time_approximate",
                "reference": config.csi_timestamp_reference,
                "mapping": (
                    "host_unix = clock_anchor.host_unix_time_s + "
                    "(usrp_time - clock_anchor.usrp_time_s)"
                ),
                "note": (
                    "The mapped host time is an approximation. Synchronize "
                    "the host/ROS clocks before joining TurtleBot mobility "
                    "data."
                ),
            },
            "csi_interpretation": {
                "shape_per_snapshot": [config.num_tx_branches],
                "description": (
                    "one complex LS estimate on one distinct pilot "
                    "subcarrier per TX branch"
                ),
                "branch_labels": list(config.branch_labels),
                "pilot_centered_bins": list(config.pilot_centered_bins),
                "pilot_rf_frequencies_hz": [
                    config.center_frequency_hz
                    + centered_bin * config.subcarrier_spacing_hz
                    for centered_bin in config.pilot_centered_bins
                ],
            },
            "storage": {
                "format": "compressed_npz_chunks",
                "raw_iq_saved": save_raw_iq,
            },
        }
        self._write_json_atomic(
            self.run_directory / "run_metadata.json",
            run_metadata,
        )
        self._write_manifest()

    def write_chunk(
        self,
        result: SoundingCSIResult,
        raw_capture: RawCapture,
        *,
        processing_buffer_first_time_s: float,
        snapshot_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Write one processed chunk and optional per-snapshot metadata.

        ``snapshot_metadata`` is the extension point for timestamp-aligned
        TurtleBot pose, velocity, or beam information.  Every value must have
        the same first dimension as ``result.csi``.
        """
        if result.num_snapshots <= 0:
            raise ValueError("cannot write an empty CSI chunk")
        if self.first_csi_timestamp_s is None:
            self.first_csi_timestamp_s = float(
                result.csi_timestamps_usrp_s[0]
            )

        dataset_row_indices = np.arange(
            self.total_snapshots,
            self.total_snapshots + result.num_snapshots,
            dtype=np.int64,
        )
        csi_period_indices = np.rint(
            (
                result.csi_timestamps_usrp_s
                - self.first_csi_timestamp_s
            )
            / self.config.csi_interval_s
        ).astype(np.int64)

        arrays: dict[str, Any] = {
            "dataset_row_indices": dataset_row_indices,
            "csi_period_indices": csi_period_indices,
            "csi": result.csi,
            "csi_repetitions": result.csi_repetitions,
            "csi_repetitions_aligned": result.csi_repetitions_aligned,
            "pilot_received": result.pilot_received,
            "transmitted_pilots": build_transmitted_pilots(self.config),
            "csi_timestamps_usrp_s": result.csi_timestamps_usrp_s,
            "csi_timestamps_host_unix_s": map_usrp_to_host_unix(
                result.csi_timestamps_usrp_s,
                self.clock_anchor,
            ),
            "pilot_timestamps_usrp_s": result.pilot_timestamps_usrp_s,
            "frame_start_sample_indices_in_processing_buffer": (
                result.frame_start_sample_indices
            ),
            "timing_error_samples": result.timing_error_samples,
            "frame_sequence_numbers_in_processing_buffer": (
                result.frame_sequence_numbers
            ),
            "zc_correlation_scores": result.zc_correlation_scores,
            "pilot_noise_power": result.pilot_noise_power,
            "pilot_snr_db": result.pilot_snr_db,
            "repetition_phase_corrections_rad": (
                result.repetition_phase_corrections_rad
            ),
            "pilot_centered_bins": np.asarray(
                self.config.pilot_centered_bins,
                dtype=np.int16,
            ),
        }
        if self.save_raw_iq:
            arrays["raw_iq"] = raw_capture.samples
            arrays["raw_iq_first_sample_time_usrp_s"] = np.float64(
                raw_capture.first_sample_time_s
            )

        if snapshot_metadata is not None:
            for name, values in snapshot_metadata.items():
                if name in arrays:
                    raise ValueError(f"snapshot metadata key already exists: {name}")
                value_array = np.asarray(values)
                if value_array.ndim == 0 or value_array.shape[0] != result.num_snapshots:
                    raise ValueError(
                        f"snapshot metadata '{name}' must have first dimension "
                        f"{result.num_snapshots}"
                    )
                arrays[name] = value_array

        chunk_name = f"csi_chunk_{self.chunk_index:06d}.npz"
        chunk_path = self.run_directory / chunk_name
        temporary_path = chunk_path.with_suffix(".npz.tmp")
        with temporary_path.open("wb") as output_file:
            np.savez_compressed(output_file, **arrays)
        temporary_path.replace(chunk_path)

        chunk_record = {
            "chunk_index": self.chunk_index,
            "file": chunk_name,
            "num_snapshots": result.num_snapshots,
            "first_dataset_row_index": int(dataset_row_indices[0]),
            "last_dataset_row_index": int(dataset_row_indices[-1]),
            "first_csi_period_index": int(csi_period_indices[0]),
            "last_csi_period_index": int(csi_period_indices[-1]),
            "first_csi_timestamp_usrp_s": float(
                result.csi_timestamps_usrp_s[0]
            ),
            "last_csi_timestamp_usrp_s": float(
                result.csi_timestamps_usrp_s[-1]
            ),
            "processing_buffer_first_time_usrp_s": (
                processing_buffer_first_time_s
            ),
            "raw_chunk_first_time_usrp_s": raw_capture.first_sample_time_s,
            "estimated_cfo_hz": result.estimated_cfo_hz,
            "mean_zc_correlation_score": float(
                np.mean(result.zc_correlation_scores)
            ),
            "mean_pilot_snr_db": float(np.mean(result.pilot_snr_db)),
        }
        self._manifest["chunks"].append(chunk_record)
        self.total_snapshots += result.num_snapshots
        self._manifest["total_snapshots"] = self.total_snapshots
        self._write_manifest()
        self.chunk_index += 1
        return chunk_path

    def _write_manifest(self) -> None:
        self._write_json_atomic(
            self.run_directory / "manifest.json",
            self._manifest,
        )

    @staticmethod
    def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2)
        temporary_path.replace(path)


def _parse_addresses(value: str) -> tuple[str, ...]:
    addresses = tuple(part.strip() for part in value.split(",") if part.strip())
    if not addresses:
        raise argparse.ArgumentTypeError("at least one TX address is required")
    return addresses


def _start_transmitter_thread(
    tx_usrp: Any,
    waveform: NDArray[np.complex64],
    start_time_s: float,
    config: SoundingConfig,
    hardware: TransmitterHardwareConfig,
    stop_event: threading.Event,
) -> tuple[threading.Thread, queue.Queue[BaseException]]:
    errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)

    def transmit() -> None:
        try:
            transmit_sounding(
                tx_usrp,
                waveform,
                start_time_s,
                config,
                hardware,
                num_periods=None,
                stop_event=stop_event,
            )
        except BaseException as error:
            errors.put(error)

    thread = threading.Thread(
        target=transmit,
        name="continuous-usrp-tx",
        daemon=True,
    )
    thread.start()
    return thread, errors


def _raise_transmitter_error(errors: queue.Queue[BaseException]) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError("continuous UHD transmitter failed") from error


def run_experiment(
    tx_hardware: TransmitterHardwareConfig,
    rx_hardware: ReceiverHardwareConfig,
    processing: ReceiverProcessingConfig,
    *,
    output_directory: str | Path,
    chunk_duration_s: float,
    duration_s: float | None,
    save_raw_iq: bool,
    check_reference_lock: bool,
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
    snapshot_metadata_provider: SnapshotMetadataProvider | None = None,
) -> Path:
    """Run synchronized continuous TX/RX and write processed CSI chunks.

    If supplied, ``snapshot_metadata_provider`` is called once per non-empty
    output chunk with ``(usrp_times, host_unix_times)``.  Its returned arrays
    are stored beside the matching CSI rows, making it suitable for
    timestamp-interpolated TurtleBot pose and velocity.
    """
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be positive")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive or None")
    chunk_samples_float = chunk_duration_s * config.sampling_rate_hz
    chunk_samples = round(chunk_samples_float)
    if not math.isclose(chunk_samples_float, chunk_samples, abs_tol=1e-9):
        raise ValueError("chunk_duration_s must contain an integer sample count")
    if chunk_samples < 2 * config.csi_interval_samples:
        raise ValueError("chunk_duration_s must span at least two CSI periods")

    tx_usrp = initialize_transmitter(
        config,
        tx_hardware,
        check_reference_lock=check_reference_lock,
        reset_hardware_time=False,
    )
    rx_usrp = initialize_receiver(
        rx_hardware,
        config,
        reset_hardware_time=False,
    )
    synchronize_devices_on_common_pps(tx_usrp, rx_usrp)
    clock_anchor = read_clock_anchor(rx_usrp)

    rx_start_time_s = choose_experiment_start_time(
        tx_usrp,
        rx_usrp,
        config.startup_lead_time_s,
    )
    # Start RX one complete period earlier so the first ZC peak is not located
    # at correlation-array index zero, which peak detectors cannot select.
    tx_start_time_s = rx_start_time_s + config.csi_interval_s
    waveform = build_sounding_period(config)
    writer = CSIChunkWriter(
        output_directory,
        config,
        tx_hardware,
        rx_hardware,
        processing,
        clock_anchor,
        save_raw_iq=save_raw_iq,
        rx_start_time_s=rx_start_time_s,
        tx_start_time_s=tx_start_time_s,
    )

    receiver = ContinuousRXCapture(
        rx_usrp,
        rx_start_time_s,
        chunk_samples,
        rx_hardware,
        config,
    )
    transmitter_stop = threading.Event()
    transmitter_thread: threading.Thread | None = None
    transmitter_errors: queue.Queue[BaseException] | None = None

    tail = np.empty(0, dtype=np.complex64)
    tail_first_time_s = rx_start_time_s
    last_saved_timestamp_s: float | None = None
    fixed_cfo_hz: float | None = None
    overlap_samples = (
        config.csi_interval_samples + config.sounding_burst_samples
    )
    target_chunks = (
        None
        if duration_s is None
        else math.ceil(duration_s / chunk_duration_s)
    )
    chunks_received = 0

    print(
        f"RX starts at UHD {rx_start_time_s:.6f} s; "
        f"TX starts at UHD {tx_start_time_s:.6f} s."
    )
    print(
        f"Processing {chunk_samples} RX samples per chunk "
        f"({chunk_duration_s:.3f} s)."
    )

    try:
        receiver.start()
        transmitter_thread, transmitter_errors = _start_transmitter_thread(
            tx_usrp,
            waveform,
            tx_start_time_s,
            config,
            tx_hardware,
            transmitter_stop,
        )

        while target_chunks is None or chunks_received < target_chunks:
            _raise_transmitter_error(transmitter_errors)
            raw_capture = receiver.get_chunk(
                timeout_s=chunk_duration_s + 5.0
            )
            _raise_transmitter_error(transmitter_errors)

            if tail.size:
                processing_samples = np.concatenate(
                    (tail, raw_capture.samples)
                )
                processing_first_time_s = tail_first_time_s
            else:
                processing_samples = raw_capture.samples
                processing_first_time_s = raw_capture.first_sample_time_s

            result = process_received_samples(
                processing_samples,
                processing_first_time_s,
                config,
                processing,
                cfo_override_hz=fixed_cfo_hz,
                cfo_phase_reference_time_s=rx_start_time_s,
            )
            if (
                processing.compensate_frequency_offset
                and fixed_cfo_hz is None
            ):
                # Reusing one estimate avoids a different phase origin and a
                # different correction slope in every saved chunk.
                fixed_cfo_hz = result.estimated_cfo_hz
            if last_saved_timestamp_s is not None:
                new_snapshot = result.csi_timestamps_usrp_s > (
                    last_saved_timestamp_s
                    + 0.5 * config.csi_interval_s
                )
                result = select_csi_result(result, new_snapshot)

            if result.num_snapshots:
                snapshot_metadata = (
                    None
                    if snapshot_metadata_provider is None
                    else snapshot_metadata_provider(
                        result.csi_timestamps_usrp_s.copy(),
                        map_usrp_to_host_unix(
                            result.csi_timestamps_usrp_s,
                            clock_anchor,
                        ),
                    )
                )
                chunk_path = writer.write_chunk(
                    result,
                    raw_capture,
                    processing_buffer_first_time_s=(
                        processing_first_time_s
                    ),
                    snapshot_metadata=snapshot_metadata,
                )
                last_saved_timestamp_s = float(
                    result.csi_timestamps_usrp_s[-1]
                )
                print(
                    f"[{chunks_received:06d}] {chunk_path.name}: "
                    f"{result.num_snapshots} CSI, "
                    f"t={result.csi_timestamps_usrp_s[0]:.6f}.."
                    f"{result.csi_timestamps_usrp_s[-1]:.6f}, "
                    f"ZC={np.mean(result.zc_correlation_scores):.3f}, "
                    f"SNR={np.mean(result.pilot_snr_db):.1f} dB"
                )
            else:
                print(f"[{chunks_received:06d}] no new complete CSI snapshot")

            retained = min(overlap_samples, processing_samples.size)
            tail = np.ascontiguousarray(
                processing_samples[-retained:],
                dtype=np.complex64,
            )
            tail_first_time_s = (
                processing_first_time_s
                + (processing_samples.size - retained)
                / config.sampling_rate_hz
            )
            chunks_received += 1
    except KeyboardInterrupt:
        print("Experiment stopped by user.")
    finally:
        transmitter_stop.set()
        receiver.close()
        if transmitter_thread is not None:
            transmitter_thread.join(timeout=5.0)
            if transmitter_thread.is_alive():
                raise RuntimeError("continuous TX worker did not stop")

    if transmitter_errors is not None:
        _raise_transmitter_error(transmitter_errors)
    print(
        f"Run complete: {writer.total_snapshots} CSI snapshots in "
        f"{writer.run_directory}"
    )
    return writer.run_directory


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tx-addresses",
        type=_parse_addresses,
        default=DEFAULT_USRP_ADDRESSES,
        help="comma-separated three-X310 transmitter IPs",
    )
    parser.add_argument(
        "--rx-address",
        required=True,
        help="TurtleBot receiver USRP IP address",
    )
    parser.add_argument("--rx-subdev", default="A:0")
    parser.add_argument("--rx-antenna", default="TX/RX")
    parser.add_argument("--rx-channel", type=int, default=0)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=10.0,
        help="experiment duration (rounded up to chunks; default: 10)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="run until Ctrl-C instead of using --duration-s",
    )
    parser.add_argument(
        "--chunk-duration-s",
        type=float,
        default=DEFAULT_SOUNDING_CONFIG.dataset_chunk_duration_s,
        help="duration saved in each dataset chunk (default: 1 second)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--compensate-cfo",
        action="store_true",
        help="estimate CFO once and apply continuous phase correction",
    )
    parser.add_argument("--save-raw-iq", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_SOUNDING_CONFIG.output_directory,
    )
    parser.add_argument(
        "--skip-ref-lock-check",
        action="store_true",
        help="skip TX ref_locked checks (not recommended)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = DEFAULT_SOUNDING_CONFIG
    tx_hardware = TransmitterHardwareConfig(addresses=args.tx_addresses)
    rx_hardware = ReceiverHardwareConfig(
        address=args.rx_address,
        rx_subdev_spec=args.rx_subdev,
        rx_antenna=args.rx_antenna,
        rx_channel=args.rx_channel,
        clock_source="external",
        time_source="external",
    )
    processing = ReceiverProcessingConfig(
        correlation_threshold=args.correlation_threshold,
        # A common OctoClock makes CFO small.  Leave correction disabled by
        # default to preserve inter-snapshot channel phase evolution.
        compensate_frequency_offset=args.compensate_cfo,
    )
    run_experiment(
        tx_hardware,
        rx_hardware,
        processing,
        output_directory=args.output_dir,
        chunk_duration_s=args.chunk_duration_s,
        duration_s=None if args.continuous else args.duration_s,
        save_raw_iq=args.save_raw_iq,
        check_reference_lock=not args.skip_ref_lock_check,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSIChunkWriter",
    "ClockAnchor",
    "SnapshotMetadataProvider",
    "choose_experiment_start_time",
    "main",
    "map_usrp_to_host_unix",
    "read_clock_anchor",
    "run_experiment",
    "select_csi_result",
    "synchronize_devices_on_common_pps",
]
