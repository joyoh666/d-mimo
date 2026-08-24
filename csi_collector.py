"""Collect per-antenna CSI for the D-MIMO prototype.

The sounding waveform follows the prototype manuscript: six transmit branches
(three RUs with two antennas each) use distinct OFDM pilot subcarriers, the UE
receives the superposed waveform on one antenna, and three consecutive pilot
symbols are averaged to produce one complex CSI sample every 5 ms.

The default invocation is a hardware-free dry run. RF streaming is enabled only
with ``--collect`` and explicit UHD device arguments::

    python csi_collector.py \
        --collect \
        --device-args "addr0=...,addr1=...,addr2=...,addr3=...,master_clock_rate=184.32e6" \
        --tx-channels 0,1,2,3,4,5 \
        --rx-channel 6 \
        --episode-id route_001 \
        --dataset-split train \
        --trajectory-id random_route_001

With two RF channels exposed per motherboard, the default mapping places the
six RU branches on motherboards 0-2 and the UE receiver on channel 0 of
motherboard 3. Check the printed mapping against ``uhd_usrp_probe`` before RF
operation. The script never transmits unless ``--collect`` is present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from config import SoundingConfig
from receiver import ReceiveResult, receive_frame
from transmitter import TransmitFrame, build_transmit_frame


Complex64Array = NDArray[np.complex64]
Float32Array = NDArray[np.float32]

SAMPLES_PER_PREDICTOR_BLOCK = 5  # 5 x 5 ms = 25 ms
SAMPLES_PER_SCHEDULER_SEGMENT = 20  # 20 x 5 ms = 100 ms
SCHEMA_VERSION = "d-mimo-per-antenna-csi-v1"


@dataclass(frozen=True)
class RuntimeConfig:
    """Hardware and episode settings that are not fixed by the manuscript."""

    device_args: str
    tx_channels: tuple[int, ...]
    rx_channel: int
    duration_s: float
    tx_gain_db: float
    rx_gain_db: float
    tx_amplitude: float
    bandwidth_hz: float
    tx_antenna: str | None
    rx_antenna: str | None
    clock_source: str
    time_source: str
    subdev_spec: str | None
    start_delay_s: float
    pre_roll_s: float
    post_roll_s: float
    reference_lock_timeout_s: float
    search_radius_samples: int
    minimum_sync_metric: float
    otw_format: str
    save_raw_iq: bool
    dataset_split: str
    trajectory_id: str | None
    environment_label: str | None
    robot_speed_mps: float | None
    notes: str | None


@dataclass(frozen=True)
class HardwareCapture:
    """One continuous UE capture and the device-time interval it represents."""

    samples: Complex64Array
    requested_rx_start_time_s: float
    actual_rx_start_time_s: float
    tx_start_time_s: float
    estimated_tx_start_host_unix_s: float
    capture_finished_host_unix_s: float
    stream_errors: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeData:
    """Model-ready CSI arrays plus alignment and measurement-quality metadata."""

    csi: Complex64Array
    per_pilot_csi: Complex64Array
    csi_gain: Float32Array
    csi_phase_rad: Float32Array
    predictor_tokens: Float32Array
    predictor_block_valid: NDArray[np.bool_]
    ru_channel_norm: Float32Array
    scheduler_segment_mean_gain: Float32Array
    scheduler_segment_valid: NDArray[np.bool_]
    pilot_snr_db: Float32Array
    elapsed_time_s: NDArray[np.float64]
    device_time_s: NDArray[np.float64]
    host_time_unix_s: NDArray[np.float64]
    sample_index: NDArray[np.int32]
    predictor_block_index: NDArray[np.int32]
    sample_in_predictor_block: NDArray[np.int8]
    scheduler_segment_index: NDArray[np.int32]
    sample_in_scheduler_segment: NDArray[np.int8]
    sync_start_sample: NDArray[np.int64]
    sync_offset_samples: NDArray[np.int32]
    sync_metric_peak: Float32Array
    cfo_estimate_hz: Float32Array
    valid: NDArray[np.bool_]
    metadata: dict[str, Any]
    raw_rx_iq: Complex64Array | None = None
    extra_arrays: dict[str, NDArray[Any]] | None = None


def parse_channels(text: str) -> tuple[int, ...]:
    """Parse a comma-separated list of distinct non-negative UHD channels."""
    try:
        channels = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Channels must be comma-separated integers."
        ) from exc
    if not channels or any(channel < 0 for channel in channels):
        raise argparse.ArgumentTypeError("Channels must be non-negative.")
    if len(set(channels)) != len(channels):
        raise argparse.ArgumentTypeError("Channels must be unique.")
    return channels


def utc_now_string() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def default_episode_id() -> str:
    return datetime.now(timezone.utc).strftime("episode_%Y%m%dT%H%M%SZ")


def make_time_indices(num_samples: int) -> dict[str, NDArray[np.integer]]:
    """Create explicit 5/25/100 ms indices so model slicing is unambiguous."""
    sample_index = np.arange(num_samples, dtype=np.int32)
    return {
        "sample_index": sample_index,
        "predictor_block_index": (
            sample_index // SAMPLES_PER_PREDICTOR_BLOCK
        ).astype(np.int32),
        "sample_in_predictor_block": (
            sample_index % SAMPLES_PER_PREDICTOR_BLOCK
        ).astype(np.int8),
        "scheduler_segment_index": (
            sample_index // SAMPLES_PER_SCHEDULER_SEGMENT
        ).astype(np.int32),
        "sample_in_scheduler_segment": (
            sample_index % SAMPLES_PER_SCHEDULER_SEGMENT
        ).astype(np.int8),
    }


def estimate_pilot_snr_db(
    result: ReceiveResult,
    cfg: SoundingConfig,
) -> Float32Array:
    """Estimate per-branch SNR using unused FFT bins as a noise-floor proxy."""
    pilot_indices = np.asarray(cfg.pilot_bins, dtype=np.int64) + cfg.nfft // 2
    noise_mask = np.ones(cfg.nfft, dtype=bool)
    noise_mask[pilot_indices] = False
    noise_mask[cfg.nfft // 2] = False  # DC

    noise_power = float(np.median(np.abs(result.received_grids[:, noise_mask]) ** 2))
    received_pilot_power = np.mean(
        np.abs(result.received_grids[:, pilot_indices]) ** 2,
        axis=0,
    )
    signal_power = np.maximum(received_pilot_power - noise_power, 0.0)
    snr = 10.0 * np.log10((signal_power + 1e-15) / (noise_power + 1e-15))
    return np.asarray(snr, dtype=np.float32)


def make_model_views(
    csi: Complex64Array,
    valid: NDArray[np.bool_],
) -> dict[str, NDArray[Any]]:
    """Create the paper's 25 ms predictor and 100 ms scheduler views."""
    if csi.ndim != 3 or csi.shape[1:] != (3, 2):
        raise ValueError("csi must have shape (time, 3, 2).")
    if valid.shape != (csi.shape[0],):
        raise ValueError("valid must have one entry per CSI sample.")

    num_predictor_blocks = csi.shape[0] // SAMPLES_PER_PREDICTOR_BLOCK
    predictor_stop = num_predictor_blocks * SAMPLES_PER_PREDICTOR_BLOCK
    predictor_csi = csi[:predictor_stop].reshape(
        num_predictor_blocks,
        SAMPLES_PER_PREDICTOR_BLOCK,
        3,
        2,
    )
    gain = np.abs(predictor_csi).transpose(0, 2, 3, 1)
    phase = np.angle(predictor_csi).transpose(0, 2, 3, 1)
    predictor_tokens = np.concatenate(
        (gain, np.cos(phase), np.sin(phase)),
        axis=-1,
    ).astype(np.float32)
    predictor_block_valid = valid[:predictor_stop].reshape(
        num_predictor_blocks,
        SAMPLES_PER_PREDICTOR_BLOCK,
    ).all(axis=1)

    ru_channel_norm = np.linalg.norm(csi, axis=2).astype(np.float32)
    num_scheduler_segments = csi.shape[0] // SAMPLES_PER_SCHEDULER_SEGMENT
    scheduler_stop = num_scheduler_segments * SAMPLES_PER_SCHEDULER_SEGMENT
    scheduler_segment_mean_gain = ru_channel_norm[:scheduler_stop].reshape(
        num_scheduler_segments,
        SAMPLES_PER_SCHEDULER_SEGMENT,
        3,
    ).mean(axis=1, dtype=np.float32)
    scheduler_segment_valid = valid[:scheduler_stop].reshape(
        num_scheduler_segments,
        SAMPLES_PER_SCHEDULER_SEGMENT,
    ).all(axis=1)

    return {
        "predictor_tokens": predictor_tokens,
        "predictor_block_valid": predictor_block_valid,
        "ru_channel_norm": ru_channel_norm,
        "scheduler_segment_mean_gain": scheduler_segment_mean_gain,
        "scheduler_segment_valid": scheduler_segment_valid,
    }


def _empty_episode_arrays(
    num_frames: int,
    cfg: SoundingConfig,
) -> dict[str, NDArray[Any]]:
    csi_shape = (num_frames, 3, 2)
    repeated_shape = (num_frames, cfg.pilot_repetitions, 3, 2)
    complex_nan = np.complex64(np.nan + 1j * np.nan)
    return {
        "csi": np.full(csi_shape, complex_nan, dtype=np.complex64),
        "per_pilot_csi": np.full(
            repeated_shape,
            complex_nan,
            dtype=np.complex64,
        ),
        "pilot_snr_db": np.full(csi_shape, np.nan, dtype=np.float32),
        "sync_start_sample": np.full(num_frames, -1, dtype=np.int64),
        "sync_offset_samples": np.full(num_frames, 0, dtype=np.int32),
        "sync_metric_peak": np.full(num_frames, np.nan, dtype=np.float32),
        "cfo_estimate_hz": np.full(num_frames, np.nan, dtype=np.float32),
        "valid": np.zeros(num_frames, dtype=bool),
    }


def estimate_episode_csi(
    received: NDArray[np.complexfloating[Any, Any]],
    tx_frame: TransmitFrame,
    cfg: SoundingConfig,
    *,
    num_frames: int,
    rx_start_time_s: float,
    tx_start_time_s: float,
    search_radius_samples: int,
    minimum_sync_metric: float,
    metadata: dict[str, Any],
    waveform_scale: float = 1.0,
    host_tx_start_unix_s: float | None = None,
    keep_raw_iq: bool = False,
) -> EpisodeData:
    """Estimate all six antenna-branch channels from a continuous UE capture.

    Each frame is searched independently around its expected 5 ms boundary.
    Failed frames remain in the timeline with NaN CSI and ``valid=False`` so a
    dropped estimate cannot silently shift later 25 ms or 100 ms windows.
    """
    cfg.validate()
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if search_radius_samples < 0:
        raise ValueError("search_radius_samples must be non-negative.")
    if not 0.0 <= minimum_sync_metric <= 1.0:
        raise ValueError("minimum_sync_metric must be in [0, 1].")
    if waveform_scale <= 0.0:
        raise ValueError("waveform_scale must be positive.")

    samples = np.asarray(received, dtype=np.complex64).reshape(-1)
    arrays = _empty_episode_arrays(num_frames, cfg)
    expected_first_start = round(
        (tx_start_time_s - rx_start_time_s) * cfg.sample_rate_hz
    )
    if expected_first_start < 0:
        raise ValueError("RX capture must start no later than TX.")

    for frame_index in range(num_frames):
        expected_start = expected_first_start + frame_index * cfg.period_samples
        window_start = max(0, expected_start - search_radius_samples)
        window_stop = min(
            samples.size,
            expected_start + search_radius_samples + cfg.burst_samples,
        )
        window = samples[window_start:window_stop]

        # The receiver needs enough room for ZC correlation and all repeated
        # pilot symbols. A truncated frame is preserved as an invalid sample.
        if window.size < cfg.burst_samples:
            continue

        try:
            result = receive_frame(
                window,
                tx_frame.zc * waveform_scale,
                tx_frame.pilot_symbols * waveform_scale,
                cfg,
            )
        except (ValueError, FloatingPointError):
            continue

        global_sync_start = window_start + result.sync_start
        sync_peak = float(np.max(result.sync_metric))
        finite = bool(
            np.all(np.isfinite(result.channel_estimates))
            and np.isfinite(result.cfo_estimate_hz)
            and np.isfinite(sync_peak)
        )
        synchronized = abs(global_sync_start - expected_start) <= search_radius_samples
        valid = finite and synchronized and sync_peak >= minimum_sync_metric

        arrays["sync_start_sample"][frame_index] = global_sync_start
        arrays["sync_offset_samples"][frame_index] = global_sync_start - expected_start
        arrays["sync_metric_peak"][frame_index] = sync_peak
        arrays["cfo_estimate_hz"][frame_index] = result.cfo_estimate_hz
        arrays["valid"][frame_index] = valid

        if valid:
            arrays["csi"][frame_index] = np.asarray(
                result.channel_estimates.reshape(3, 2),
                dtype=np.complex64,
            )
            arrays["per_pilot_csi"][frame_index] = np.asarray(
                result.per_pilot_channel_estimates.reshape(
                    cfg.pilot_repetitions,
                    3,
                    2,
                ),
                dtype=np.complex64,
            )
            arrays["pilot_snr_db"][frame_index] = estimate_pilot_snr_db(
                result,
                cfg,
            ).reshape(3, 2)

    indices = make_time_indices(num_frames)
    elapsed_time_s = (
        indices["sample_index"].astype(np.float64) * cfg.pilot_period_s
    )
    device_time_s = tx_start_time_s + elapsed_time_s
    if host_tx_start_unix_s is None:
        host_time_unix_s = np.full(num_frames, np.nan, dtype=np.float64)
    else:
        host_time_unix_s = host_tx_start_unix_s + elapsed_time_s
    csi = np.asarray(arrays["csi"], dtype=np.complex64)
    model_views = make_model_views(
        csi,
        np.asarray(arrays["valid"], dtype=bool),
    )

    return EpisodeData(
        csi=csi,
        per_pilot_csi=np.asarray(arrays["per_pilot_csi"], dtype=np.complex64),
        csi_gain=np.asarray(np.abs(csi), dtype=np.float32),
        csi_phase_rad=np.asarray(np.angle(csi), dtype=np.float32),
        predictor_tokens=np.asarray(
            model_views["predictor_tokens"], dtype=np.float32
        ),
        predictor_block_valid=np.asarray(
            model_views["predictor_block_valid"], dtype=bool
        ),
        ru_channel_norm=np.asarray(
            model_views["ru_channel_norm"], dtype=np.float32
        ),
        scheduler_segment_mean_gain=np.asarray(
            model_views["scheduler_segment_mean_gain"], dtype=np.float32
        ),
        scheduler_segment_valid=np.asarray(
            model_views["scheduler_segment_valid"], dtype=bool
        ),
        pilot_snr_db=np.asarray(arrays["pilot_snr_db"], dtype=np.float32),
        elapsed_time_s=elapsed_time_s,
        device_time_s=device_time_s,
        host_time_unix_s=host_time_unix_s,
        sample_index=np.asarray(indices["sample_index"], dtype=np.int32),
        predictor_block_index=np.asarray(
            indices["predictor_block_index"], dtype=np.int32
        ),
        sample_in_predictor_block=np.asarray(
            indices["sample_in_predictor_block"], dtype=np.int8
        ),
        scheduler_segment_index=np.asarray(
            indices["scheduler_segment_index"], dtype=np.int32
        ),
        sample_in_scheduler_segment=np.asarray(
            indices["sample_in_scheduler_segment"], dtype=np.int8
        ),
        sync_start_sample=np.asarray(arrays["sync_start_sample"], dtype=np.int64),
        sync_offset_samples=np.asarray(
            arrays["sync_offset_samples"], dtype=np.int32
        ),
        sync_metric_peak=np.asarray(arrays["sync_metric_peak"], dtype=np.float32),
        cfo_estimate_hz=np.asarray(arrays["cfo_estimate_hz"], dtype=np.float32),
        valid=np.asarray(arrays["valid"], dtype=bool),
        metadata=metadata,
        raw_rx_iq=samples if keep_raw_iq else None,
    )


def make_metadata(
    *,
    cfg: SoundingConfig,
    runtime: RuntimeConfig,
    episode_id: str,
    capture: HardwareCapture | None,
    output_path: Path,
) -> dict[str, Any]:
    """Build self-describing metadata for training and reproducibility."""
    num_frames = round(runtime.duration_s / cfg.pilot_period_s)
    branch_map = []
    for branch, (tx_channel, pilot_bin) in enumerate(
        zip(runtime.tx_channels, cfg.pilot_bins)
    ):
        branch_map.append(
            {
                "branch_index": branch,
                "ru_index": branch // 2,
                "ru_label": f"RU{branch // 2 + 1}",
                "antenna_index": branch % 2,
                "antenna_label": f"ANT{branch % 2 + 1}",
                "uhd_tx_channel": tx_channel,
                "pilot_centered_bin": pilot_bin,
                "pilot_offset_hz": pilot_bin * cfg.subcarrier_spacing_hz,
                "pilot_frequency_hz": (
                    cfg.center_frequency_hz
                    + pilot_bin * cfg.subcarrier_spacing_hz
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now_string(),
        "episode_id": episode_id,
        "output_path": str(output_path.resolve()),
        "source": {
            "prototype": (
                "Learning-Based Predictive Phase Alignment and RU Scheduling "
                "for Distributed MIMO"
            ),
            "csi_definition": (
                "Complex least-squares RU-antenna-to-UE channel coefficient; "
                "three consecutive pilots averaged coherently."
            ),
            "paper_fixed_parameters": [
                "3 RUs x 2 antennas",
                "single-antenna UE",
                "2.2 GHz carrier",
                "1.4 MHz nominal bandwidth",
                "15 kHz subcarrier spacing",
                "FFT size 128",
                "5 ms CSI interval",
                "distinct pilot subcarrier per transmit branch",
                "3-pilot complex averaging",
            ],
            "implementation_assumptions": [
                "CP length, ZC length/root, and pilot-bin positions are not "
                "specified by the manuscript and come from SoundingConfig.",
                "Only sync_tx_branch transmits the Zadoff-Chu timing sequence.",
                "The six nearby pilot subcarriers are treated as samples of a "
                "flat narrowband channel, as in the manuscript.",
            ],
            "rf_chain_calibration_applied": False,
            "calibration_note": (
                "Stored CSI includes propagation and uncalibrated TX/RX RF-chain "
                "responses. Apply a separately measured complex calibration "
                "coefficient if propagation-only CSI is required."
            ),
        },
        "array_axes": {
            "csi": ["csi_sample", "ru", "antenna"],
            "per_pilot_csi": [
                "csi_sample",
                "pilot_repetition",
                "ru",
                "antenna",
            ],
            "predictor_tokens": [
                "predictor_block",
                "ru",
                "antenna",
                "gain_5_then_cos_phase_5_then_sin_phase_5",
            ],
            "ru_channel_norm": ["csi_sample", "ru"],
            "scheduler_segment_mean_gain": ["scheduler_segment", "ru"],
            "pilot_snr_db": ["csi_sample", "ru", "antenna"],
        },
        "model_alignment": {
            "csi_interval_ms": cfg.pilot_period_s * 1e3,
            "predictor_block_samples": SAMPLES_PER_PREDICTOR_BLOCK,
            "predictor_block_ms": (
                SAMPLES_PER_PREDICTOR_BLOCK * cfg.pilot_period_s * 1e3
            ),
            "scheduler_segment_samples": SAMPLES_PER_SCHEDULER_SEGMENT,
            "scheduler_segment_ms": (
                SAMPLES_PER_SCHEDULER_SEGMENT * cfg.pilot_period_s * 1e3
            ),
            "recommended_predictor_context_blocks_min": 10,
            "recommended_predictor_context_ms_min": 250.0,
            "paper_prediction_horizon_blocks": 5,
            "paper_prediction_horizon_ms": 125.0,
        },
        "sounding_config": asdict(cfg),
        "runtime_config": asdict(runtime),
        "collection": {
            "requested_duration_s": runtime.duration_s,
            "num_csi_samples": num_frames,
            "episode_duration_s": num_frames * cfg.pilot_period_s,
            "rx_channel": runtime.rx_channel,
            "raw_iq_saved": runtime.save_raw_iq,
            "requested_rx_start_device_time_s": (
                None if capture is None else capture.requested_rx_start_time_s
            ),
            "actual_rx_start_device_time_s": (
                None if capture is None else capture.actual_rx_start_time_s
            ),
            "tx_start_device_time_s": (
                None if capture is None else capture.tx_start_time_s
            ),
            "estimated_tx_start_host_utc": (
                None
                if capture is None
                else datetime.fromtimestamp(
                    capture.estimated_tx_start_host_unix_s,
                    tz=timezone.utc,
                ).isoformat(timespec="milliseconds")
            ),
            "estimated_tx_start_host_unix_s": (
                None
                if capture is None
                else capture.estimated_tx_start_host_unix_s
            ),
            "capture_finished_host_unix_s": (
                None if capture is None else capture.capture_finished_host_unix_s
            ),
            "host_timestamp_note": (
                "Host timestamps are approximate unless the host clock is "
                "disciplined to the same reference as the USRPs."
            ),
            "stream_errors": [] if capture is None else list(capture.stream_errors),
        },
        "episode_labels": {
            "dataset_split": runtime.dataset_split,
            "trajectory_id": runtime.trajectory_id,
            "environment_label": runtime.environment_label,
            "robot_speed_mps": runtime.robot_speed_mps,
            "notes": runtime.notes,
        },
        "branch_map": branch_map,
        "quality_fields": {
            "valid": (
                "True only when the frame is complete, finite, near the "
                "expected boundary, and above minimum_sync_metric."
            ),
            "sync_offset_samples": "Detected minus expected frame start.",
            "sync_metric_peak": "Peak normalized Zadoff-Chu correlation.",
            "cfo_estimate_hz": "Per-frame repeated-pilot CFO estimate.",
            "pilot_snr_db": "Unused-subcarrier noise-floor estimate.",
        },
    }


def save_episode(path: Path, episode: EpisodeData) -> None:
    """Atomically save one episode as a compressed, self-describing NPZ."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    arrays: dict[str, Any] = {
        "csi": episode.csi,
        "per_pilot_csi": episode.per_pilot_csi,
        "csi_gain": episode.csi_gain,
        "csi_phase_rad": episode.csi_phase_rad,
        "predictor_tokens": episode.predictor_tokens,
        "predictor_block_valid": episode.predictor_block_valid,
        "ru_channel_norm": episode.ru_channel_norm,
        "scheduler_segment_mean_gain": episode.scheduler_segment_mean_gain,
        "scheduler_segment_valid": episode.scheduler_segment_valid,
        "pilot_snr_db": episode.pilot_snr_db,
        "elapsed_time_s": episode.elapsed_time_s,
        "device_time_s": episode.device_time_s,
        "host_time_unix_s": episode.host_time_unix_s,
        "sample_index": episode.sample_index,
        "predictor_block_index": episode.predictor_block_index,
        "sample_in_predictor_block": episode.sample_in_predictor_block,
        "scheduler_segment_index": episode.scheduler_segment_index,
        "sample_in_scheduler_segment": episode.sample_in_scheduler_segment,
        "sync_start_sample": episode.sync_start_sample,
        "sync_offset_samples": episode.sync_offset_samples,
        "sync_metric_peak": episode.sync_metric_peak,
        "cfo_estimate_hz": episode.cfo_estimate_hz,
        "valid": episode.valid,
        "metadata_json": np.asarray(
            json.dumps(episode.metadata, ensure_ascii=False, sort_keys=True)
        ),
    }
    if episode.raw_rx_iq is not None:
        arrays["raw_rx_iq"] = episode.raw_rx_iq
    if episode.extra_arrays is not None:
        reserved = set(arrays).intersection(episode.extra_arrays)
        if reserved:
            raise ValueError(
                "extra_arrays contains reserved field(s): "
                + ", ".join(sorted(reserved))
            )
        for name, value in episode.extra_arrays.items():
            if not name or not name.isidentifier():
                raise ValueError(
                    f"Invalid extra array name {name!r}; use a Python identifier."
                )
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(
                    f"extra_arrays[{name!r}] must not use object dtype."
                )
            arrays[name] = array

    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class _FiniteReceiver(threading.Thread):
    def __init__(
        self,
        *,
        uhd: Any,
        streamer: Any,
        num_samples: int,
        start_time_s: float,
    ) -> None:
        super().__init__(name="d-mimo-rx", daemon=True)
        self.uhd = uhd
        self.streamer = streamer
        self.num_samples = num_samples
        self.start_time_s = start_time_s
        self.samples = np.zeros(num_samples, dtype=np.complex64)
        self.actual_start_time_s = start_time_s
        self.errors: list[str] = []
        self.exception: BaseException | None = None

    def run(self) -> None:
        try:
            command = self.uhd.types.StreamCMD(
                self.uhd.types.StreamMode.num_done
            )
            command.stream_now = False
            command.time_spec = self.uhd.types.TimeSpec(self.start_time_s)
            command.num_samps = self.num_samples
            self.streamer.issue_stream_cmd(command)

            metadata = self.uhd.types.RXMetadata()
            max_samples = int(self.streamer.get_max_num_samps())
            buffer = np.zeros((1, max_samples), dtype=np.complex64)
            cursor = 0
            first_packet = True
            consecutive_timeouts = 0

            while cursor < self.num_samples:
                count = int(self.streamer.recv(buffer, metadata, 2.0))
                error_code = metadata.error_code
                if error_code != self.uhd.types.RXMetadataErrorCode.none:
                    message = str(error_code)
                    self.errors.append(message)
                    if error_code == self.uhd.types.RXMetadataErrorCode.timeout:
                        consecutive_timeouts += 1
                        if consecutive_timeouts >= 3:
                            raise RuntimeError(
                                "RX timed out three consecutive times."
                            )
                        continue
                    raise RuntimeError(f"RX metadata error: {message}")
                else:
                    consecutive_timeouts = 0

                if count <= 0:
                    continue
                if first_packet and getattr(metadata, "has_time_spec", False):
                    self.actual_start_time_s = float(
                        metadata.time_spec.get_real_secs()
                    )
                first_packet = False
                stop = min(cursor + count, self.num_samples)
                self.samples[cursor:stop] = buffer[0, : stop - cursor]
                cursor = stop

            if cursor != self.num_samples:
                raise RuntimeError(
                    f"RX expected {self.num_samples} samples, received {cursor}."
                )
        except BaseException as exc:  # propagated after both threads join
            self.exception = exc


class _RepeatedTransmitter(threading.Thread):
    def __init__(
        self,
        *,
        uhd: Any,
        streamer: Any,
        frame: Complex64Array,
        num_frames: int,
        start_time_s: float,
    ) -> None:
        super().__init__(name="d-mimo-tx", daemon=True)
        self.uhd = uhd
        self.streamer = streamer
        self.frame = frame
        self.num_frames = num_frames
        self.start_time_s = start_time_s
        self.exception: BaseException | None = None

    def run(self) -> None:
        try:
            metadata = self.uhd.types.TXMetadata()
            metadata.has_time_spec = True
            metadata.time_spec = self.uhd.types.TimeSpec(self.start_time_s)
            metadata.start_of_burst = True

            for _ in range(self.num_frames):
                cursor = 0
                while cursor < self.frame.shape[1]:
                    count = int(self.streamer.send(self.frame[:, cursor:], metadata))
                    if count <= 0:
                        raise RuntimeError("TX streamer accepted zero samples.")
                    cursor += count
                    metadata.has_time_spec = False
                    metadata.start_of_burst = False

            metadata.end_of_burst = True
            self.streamer.send(
                np.zeros((self.frame.shape[0], 0), dtype=np.complex64),
                metadata,
            )
        except BaseException as exc:  # propagated after both threads join
            self.exception = exc


def import_uhd() -> Any:
    try:
        import uhd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The UHD Python module is unavailable. Run this script with the "
            "Python environment that contains pyuhd."
        ) from exc
    return uhd


def _wait_for_reference_lock(usrp: Any, timeout_s: float) -> None:
    pending: set[int] = set()
    for motherboard in range(int(usrp.get_num_mboards())):
        names = set(usrp.get_mboard_sensor_names(motherboard))
        if "ref_locked" in names:
            pending.add(motherboard)

    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        for motherboard in tuple(pending):
            if usrp.get_mboard_sensor("ref_locked", motherboard).to_bool():
                pending.remove(motherboard)
        if pending:
            time.sleep(0.1)
    if pending:
        raise RuntimeError(
            "External reference did not lock on motherboard(s): "
            + ", ".join(str(index) for index in sorted(pending))
        )


def prepare_hardware(
    uhd: Any,
    cfg: SoundingConfig,
    runtime: RuntimeConfig,
) -> tuple[Any, Any, Any]:
    """Open MultiUSRP, synchronize it, and create six-TX/one-RX streamers."""
    usrp = uhd.usrp.MultiUSRP(runtime.device_args)
    num_mboards = int(usrp.get_num_mboards())
    if runtime.subdev_spec:
        spec = uhd.usrp.SubdevSpec(runtime.subdev_spec)
        for motherboard in range(num_mboards):
            usrp.set_tx_subdev_spec(spec, motherboard)
            usrp.set_rx_subdev_spec(spec, motherboard)

    for motherboard in range(num_mboards):
        usrp.set_clock_source(runtime.clock_source, motherboard)
        usrp.set_time_source(runtime.time_source, motherboard)
    if runtime.clock_source != "internal":
        _wait_for_reference_lock(usrp, runtime.reference_lock_timeout_s)

    # set_time_unknown_pps applies TimeSpec(0) at the next PPS edge. Waiting
    # afterward prevents scheduled commands from being constructed mid-update.
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    time.sleep(1.1)

    available_tx = int(usrp.get_tx_num_channels())
    available_rx = int(usrp.get_rx_num_channels())
    if any(channel >= available_tx for channel in runtime.tx_channels):
        raise RuntimeError(
            f"Requested TX {runtime.tx_channels}, but UHD exposes "
            f"{available_tx} channel(s)."
        )
    if runtime.rx_channel >= available_rx:
        raise RuntimeError(
            f"Requested RX {runtime.rx_channel}, but UHD exposes "
            f"{available_rx} channel(s)."
        )

    for channel in runtime.tx_channels:
        usrp.set_tx_rate(cfg.sample_rate_hz, channel)
        usrp.set_tx_freq(
            uhd.libpyuhd.types.tune_request(cfg.center_frequency_hz),
            channel,
        )
        usrp.set_tx_gain(runtime.tx_gain_db, channel)
        usrp.set_tx_bandwidth(runtime.bandwidth_hz, channel)
        if runtime.tx_antenna:
            usrp.set_tx_antenna(runtime.tx_antenna, channel)

    usrp.set_rx_rate(cfg.sample_rate_hz, runtime.rx_channel)
    usrp.set_rx_freq(
        uhd.libpyuhd.types.tune_request(cfg.center_frequency_hz),
        runtime.rx_channel,
    )
    usrp.set_rx_gain(runtime.rx_gain_db, runtime.rx_channel)
    usrp.set_rx_bandwidth(runtime.bandwidth_hz, runtime.rx_channel)
    usrp.set_rx_dc_offset(False, runtime.rx_channel)
    if runtime.rx_antenna:
        usrp.set_rx_antenna(runtime.rx_antenna, runtime.rx_channel)

    for channel in runtime.tx_channels:
        actual = float(usrp.get_tx_rate(channel))
        if abs(actual - cfg.sample_rate_hz) / cfg.sample_rate_hz > 1e-6:
            raise RuntimeError(
                f"TX channel {channel} rate was coerced to {actual} S/s."
            )
    actual_rx_rate = float(usrp.get_rx_rate(runtime.rx_channel))
    if abs(actual_rx_rate - cfg.sample_rate_hz) / cfg.sample_rate_hz > 1e-6:
        raise RuntimeError(
            f"RX channel {runtime.rx_channel} rate was coerced to "
            f"{actual_rx_rate} S/s."
        )

    tx_args = uhd.usrp.StreamArgs("fc32", runtime.otw_format)
    tx_args.channels = list(runtime.tx_channels)
    rx_args = uhd.usrp.StreamArgs("fc32", runtime.otw_format)
    rx_args.channels = [runtime.rx_channel]
    return usrp, usrp.get_tx_stream(tx_args), usrp.get_rx_stream(rx_args)


def transmit_scale(tx_frame: TransmitFrame, amplitude: float) -> float:
    if not 0.0 < amplitude <= 1.0:
        raise ValueError("tx_amplitude must be in (0, 1].")
    peak = float(np.max(np.abs(tx_frame.branch_samples)))
    if peak <= 0.0:
        raise ValueError("Generated transmit waveform is empty.")
    return amplitude / peak


def prepare_tx_iq(tx_frame: TransmitFrame, amplitude: float) -> Complex64Array:
    scale = transmit_scale(tx_frame, amplitude)
    return np.ascontiguousarray(
        tx_frame.branch_samples * scale,
        dtype=np.complex64,
    )


def collect_hardware_episode(
    cfg: SoundingConfig,
    runtime: RuntimeConfig,
    tx_frame: TransmitFrame,
) -> HardwareCapture:
    """Continuously stream a repeated 5 ms frame and capture the UE channel."""
    uhd = import_uhd()
    usrp, tx_streamer, rx_streamer = prepare_hardware(uhd, cfg, runtime)
    tx_iq = prepare_tx_iq(tx_frame, runtime.tx_amplitude)
    num_frames = round(runtime.duration_s / cfg.pilot_period_s)
    current_device_time_s = float(usrp.get_time_now().get_real_secs())
    current_host_time_s = time.time()
    tx_start_time_s = current_device_time_s + runtime.start_delay_s
    estimated_tx_start_host_unix_s = (
        current_host_time_s + runtime.start_delay_s
    )
    requested_rx_start_time_s = tx_start_time_s - runtime.pre_roll_s
    num_rx_samples = round(
        (runtime.pre_roll_s + num_frames * cfg.pilot_period_s + runtime.post_roll_s)
        * cfg.sample_rate_hz
    )

    receiver = _FiniteReceiver(
        uhd=uhd,
        streamer=rx_streamer,
        num_samples=num_rx_samples,
        start_time_s=requested_rx_start_time_s,
    )
    transmitter = _RepeatedTransmitter(
        uhd=uhd,
        streamer=tx_streamer,
        frame=tx_iq,
        num_frames=num_frames,
        start_time_s=tx_start_time_s,
    )
    receiver.start()
    transmitter.start()
    transmitter.join()
    receiver.join()

    if transmitter.exception is not None:
        raise RuntimeError("TX streaming failed.") from transmitter.exception
    if receiver.exception is not None:
        raise RuntimeError("RX streaming failed.") from receiver.exception

    return HardwareCapture(
        samples=receiver.samples,
        requested_rx_start_time_s=requested_rx_start_time_s,
        actual_rx_start_time_s=receiver.actual_start_time_s,
        tx_start_time_s=tx_start_time_s,
        estimated_tx_start_host_unix_s=estimated_tx_start_host_unix_s,
        capture_finished_host_unix_s=time.time(),
        stream_errors=tuple(receiver.errors),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect six per-antenna D-MIMO CSI streams. Without --collect, "
            "validate and print the plan without opening a USRP."
        )
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--device-args", default="")
    parser.add_argument(
        "--tx-channels",
        type=parse_channels,
        default=parse_channels("0,1,2,3,4,5"),
    )
    parser.add_argument("--rx-channel", type=int, default=6)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument(
        "--dataset-split",
        choices=("train", "validation", "test", "unspecified"),
        default="unspecified",
    )
    parser.add_argument("--trajectory-id", default=None)
    parser.add_argument("--environment-label", default=None)
    parser.add_argument("--robot-speed-mps", type=float, default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tx-gain-db", type=float, default=15.0)
    parser.add_argument("--rx-gain-db", type=float, default=30.0)
    parser.add_argument("--tx-amplitude", type=float, default=0.20)
    parser.add_argument("--bandwidth-hz", type=float, default=1.4e6)
    parser.add_argument("--tx-antenna", default="TX/RX")
    parser.add_argument("--rx-antenna", default="RX2")
    parser.add_argument("--clock-source", default="external")
    parser.add_argument("--time-source", default="external")
    parser.add_argument("--subdev-spec", default="A:0 B:0")
    parser.add_argument("--start-delay-s", type=float, default=1.0)
    parser.add_argument("--pre-roll-ms", type=float, default=10.0)
    parser.add_argument("--post-roll-ms", type=float, default=10.0)
    parser.add_argument("--reference-lock-timeout-s", type=float, default=10.0)
    parser.add_argument("--search-radius-samples", type=int, default=64)
    parser.add_argument("--minimum-sync-metric", type=float, default=0.05)
    parser.add_argument("--otw-format", choices=("sc16", "sc8"), default="sc16")
    parser.add_argument(
        "--save-raw-iq",
        action="store_true",
        help="Also include the large continuous UE IQ capture in the NPZ.",
    )
    return parser


def validate_runtime(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    cfg: SoundingConfig,
) -> RuntimeConfig:
    if args.collect and not args.device_args:
        parser.error("--device-args is required with --collect.")
    if len(args.tx_channels) != cfg.num_tx_branches:
        parser.error(f"Exactly {cfg.num_tx_branches} TX channels are required.")
    if args.rx_channel < 0:
        parser.error("--rx-channel must be non-negative.")
    if args.rx_channel in args.tx_channels:
        parser.error("The UE RX channel must not also be an RU TX channel.")
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive.")
    if args.robot_speed_mps is not None and args.robot_speed_mps < 0.0:
        parser.error("--robot-speed-mps must be non-negative.")
    num_frames = round(args.duration_s / cfg.pilot_period_s)
    if not np.isclose(
        num_frames * cfg.pilot_period_s,
        args.duration_s,
        atol=1e-12,
        rtol=0.0,
    ):
        parser.error("--duration-s must be an integer multiple of 5 ms.")
    if not 0.0 < args.tx_amplitude <= 1.0:
        parser.error("--tx-amplitude must be in (0, 1].")
    if args.start_delay_s <= args.pre_roll_ms / 1e3:
        parser.error("--start-delay-s must be greater than pre-roll.")
    if args.pre_roll_ms < 0.0 or args.post_roll_ms < 0.0:
        parser.error("Pre-roll and post-roll must be non-negative.")
    if args.reference_lock_timeout_s <= 0.0:
        parser.error("--reference-lock-timeout-s must be positive.")
    if args.search_radius_samples < 0:
        parser.error("--search-radius-samples must be non-negative.")
    if not 0.0 <= args.minimum_sync_metric <= 1.0:
        parser.error("--minimum-sync-metric must be in [0, 1].")

    return RuntimeConfig(
        device_args=args.device_args,
        tx_channels=args.tx_channels,
        rx_channel=args.rx_channel,
        duration_s=args.duration_s,
        tx_gain_db=args.tx_gain_db,
        rx_gain_db=args.rx_gain_db,
        tx_amplitude=args.tx_amplitude,
        bandwidth_hz=args.bandwidth_hz,
        tx_antenna=args.tx_antenna.strip() or None,
        rx_antenna=args.rx_antenna.strip() or None,
        clock_source=args.clock_source,
        time_source=args.time_source,
        subdev_spec=args.subdev_spec.strip() or None,
        start_delay_s=args.start_delay_s,
        pre_roll_s=args.pre_roll_ms / 1e3,
        post_roll_s=args.post_roll_ms / 1e3,
        reference_lock_timeout_s=args.reference_lock_timeout_s,
        search_radius_samples=args.search_radius_samples,
        minimum_sync_metric=args.minimum_sync_metric,
        otw_format=args.otw_format,
        save_raw_iq=args.save_raw_iq,
        dataset_split=args.dataset_split,
        trajectory_id=(args.trajectory_id or "").strip() or None,
        environment_label=(args.environment_label or "").strip() or None,
        robot_speed_mps=args.robot_speed_mps,
        notes=(args.notes or "").strip() or None,
    )


def print_plan(
    cfg: SoundingConfig,
    runtime: RuntimeConfig,
    episode_id: str,
    output_path: Path,
) -> None:
    num_frames = round(runtime.duration_s / cfg.pilot_period_s)
    print("=== Per-antenna D-MIMO CSI collection plan ===")
    print(f"Episode            : {episode_id}")
    print(f"Dataset split      : {runtime.dataset_split}")
    print(f"Trajectory         : {runtime.trajectory_id or '(not set)'}")
    print(f"Output             : {output_path.resolve()}")
    print(f"Device args        : {runtime.device_args or '(dry run / not set)'}")
    print(f"RU TX channels     : {runtime.tx_channels}")
    print(f"UE RX channel      : {runtime.rx_channel}")
    print(f"CSI tensor         : ({num_frames}, 3, 2) complex64")
    print(f"CSI interval       : {cfg.pilot_period_s * 1e3:.1f} ms")
    print(f"Predictor blocks   : {num_frames // 5} x 25 ms")
    print(f"Scheduler segments : {num_frames // 20} x 100 ms")
    print(f"Duration           : {num_frames * cfg.pilot_period_s:.3f} s")
    print(
        f"Waveform           : {cfg.center_frequency_hz / 1e9:.3f} GHz, "
        f"{cfg.nominal_bandwidth_hz / 1e6:.1f} MHz, "
        f"NFFT={cfg.nfft}, SCS={cfg.subcarrier_spacing_hz / 1e3:.1f} kHz"
    )
    print(f"Pilot bins         : {cfg.pilot_bins}")
    print(f"Pilots / estimate  : {cfg.pilot_repetitions}")
    for branch, (channel, pilot_bin) in enumerate(
        zip(runtime.tx_channels, cfg.pilot_bins)
    ):
        print(
            f"  branch {branch}: RU{branch // 2 + 1}/ANT{branch % 2 + 1} "
            f"-> TX ch {channel}, pilot bin {pilot_bin:+d}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = SoundingConfig()
    cfg.validate()
    runtime = validate_runtime(parser, args, cfg)
    episode_id = args.episode_id or default_episode_id()
    output_path = args.output or Path("output/csi") / f"{episode_id}.npz"
    tx_frame = build_transmit_frame(cfg)
    prepare_tx_iq(tx_frame, runtime.tx_amplitude)  # waveform validation
    print_plan(cfg, runtime, episode_id, output_path)

    if not args.collect:
        print("Dry run complete. No USRP was opened and no RF was transmitted.")
        return 0

    capture = collect_hardware_episode(cfg, runtime, tx_frame)
    metadata = make_metadata(
        cfg=cfg,
        runtime=runtime,
        episode_id=episode_id,
        capture=capture,
        output_path=output_path,
    )
    num_frames = round(runtime.duration_s / cfg.pilot_period_s)
    episode = estimate_episode_csi(
        capture.samples,
        tx_frame,
        cfg,
        num_frames=num_frames,
        rx_start_time_s=capture.actual_rx_start_time_s,
        tx_start_time_s=capture.tx_start_time_s,
        search_radius_samples=runtime.search_radius_samples,
        minimum_sync_metric=runtime.minimum_sync_metric,
        metadata=metadata,
        waveform_scale=transmit_scale(tx_frame, runtime.tx_amplitude),
        host_tx_start_unix_s=capture.estimated_tx_start_host_unix_s,
        keep_raw_iq=runtime.save_raw_iq,
    )
    episode.metadata["collection"]["valid_csi_samples"] = int(
        np.count_nonzero(episode.valid)
    )
    episode.metadata["collection"]["invalid_csi_samples"] = int(
        episode.valid.size - np.count_nonzero(episode.valid)
    )
    save_episode(output_path, episode)

    valid_count = int(np.count_nonzero(episode.valid))
    print(
        f"Saved {valid_count}/{episode.valid.size} valid CSI samples to "
        f"{output_path.resolve()}"
    )
    if capture.stream_errors:
        print(
            "Warning: UHD reported RX metadata errors; inspect stream_errors "
            "and valid in the saved metadata.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("CSI collection interrupted.", file=sys.stderr)
        raise SystemExit(130)
