"""Generate and transmit the six-branch 5 ms sounding waveform.

The default deployment mirrors the original three-X310 transmitter:

* 192.168.110.2, 192.168.10.2, and 192.168.100.2
* two TX channels (A:0 and B:0) per X310
* a common 10 MHz reference and PPS supplied by an OctoClock

The waveform layout is the explicit implementation assumption in
``sounding_config.py``: a standalone time-domain Zadoff-Chu sequence, a zero
guard, three CP-OFDM pilot symbols, and zero padding up to exactly 5 ms.

Run ``python sounding_transmitter.py --dry-run`` without UHD hardware to
inspect and validate the generated samples.  Actual RF transmission requires
Ettus UHD Python bindings and correctly connected OctoClock outputs.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from threading import Event
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

if __package__:  # Package import from the repository root.
    from .helper_function import get_zc_sequence, ofdm_modulate_symbol
    from .modulate import modulate
    from .sounding_config import DEFAULT_SOUNDING_CONFIG, SoundingConfig
else:  # Direct execution from inside USRP/.
    from helper_function import get_zc_sequence, ofdm_modulate_symbol
    from modulate import modulate
    from sounding_config import DEFAULT_SOUNDING_CONFIG, SoundingConfig


ComplexArray = NDArray[np.complex64]

DEFAULT_USRP_ADDRESSES = (
    "192.168.110.2",
    "192.168.10.2",
    "192.168.100.2",
)


@dataclass(frozen=True, slots=True)
class TransmitterHardwareConfig:
    """Deployment-specific settings for the three synchronized X310s."""

    addresses: tuple[str, ...] = DEFAULT_USRP_ADDRESSES
    tx_subdev_spec: str = "A:0 B:0"
    tx_antenna: str = "TX/RX"
    reference_lock_timeout_s: float = 10.0
    periods_per_stream_chunk: int = 20

    def validate(self, sounding: SoundingConfig) -> None:
        if len(self.addresses) != sounding.num_rus:
            raise ValueError(
                f"expected {sounding.num_rus} USRP addresses, "
                f"got {len(self.addresses)}"
            )
        if any(not address.strip() for address in self.addresses):
            raise ValueError("USRP addresses must not be empty")
        if self.reference_lock_timeout_s <= 0:
            raise ValueError("reference_lock_timeout_s must be positive")
        if self.periods_per_stream_chunk <= 0:
            raise ValueError("periods_per_stream_chunk must be positive")

    def device_args(self, sounding: SoundingConfig) -> str:
        """Return UHD arguments for one synchronized MultiUSRP object."""
        self.validate(sounding)
        address_args = [
            f"addr{index}={address}"
            for index, address in enumerate(self.addresses)
        ]
        address_args.append(
            f"master_clock_rate={sounding.master_clock_rate_hz:.0f}"
        )
        return ",".join(address_args)


DEFAULT_HARDWARE_CONFIG = TransmitterHardwareConfig()


def _centered_bin_to_active_index(
    centered_bin: int,
    num_active_subcarriers: int,
) -> int:
    """Map ``[-N/2, ..., -1, +1, ..., +N/2]`` to an array index."""
    half = num_active_subcarriers // 2
    if centered_bin == 0 or centered_bin < -half or centered_bin > half:
        raise ValueError("centered_bin is outside the non-DC active allocation")
    if centered_bin < 0:
        return centered_bin + half
    return half + centered_bin - 1


def build_transmitted_pilots(
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
) -> ComplexArray:
    """Return known frequency-domain pilots with shape ``(repeat, branch)``.

    The QPSK values are deterministic from ``pilot_seed``.  Multiplication by
    ``sqrt(fft_size)`` makes each branch's single-tone time-domain envelope
    equal to ``digital_power_scale`` when the unitary IFFT is used.  These are
    the exact complex reference values that the receiver should use for LS
    channel estimation.
    """
    if config.pilot_modulation.lower() != "qpsk":
        raise ValueError("only QPSK sounding pilots are currently implemented")

    rng = np.random.default_rng(config.pilot_seed)
    bits = rng.integers(
        0,
        2,
        size=config.pilot_repetitions * config.num_tx_branches * 2,
        dtype=np.uint8,
    )
    qpsk = modulate(bits, modulation_order=4).reshape(
        config.pilot_repetitions,
        config.num_tx_branches,
    )
    frequency_domain_scale = (
        np.sqrt(config.fft_size) * config.digital_power_scale
    )
    return (qpsk * frequency_domain_scale).astype(np.complex64)


def build_sounding_period(
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
) -> ComplexArray:
    """Build one 5 ms TX period with shape ``(branch, time_sample)``."""
    waveform = np.zeros(
        (config.num_tx_branches, config.csi_interval_samples),
        dtype=np.complex64,
    )

    zc = get_zc_sequence(config.zc_length_samples, config.zc_root)
    waveform[
        config.sync_tx_branch,
        : config.zc_length_samples,
    ] = zc * config.digital_power_scale

    transmitted_pilots = build_transmitted_pilots(config)
    for repetition in range(config.pilot_repetitions):
        active = np.zeros(
            (config.num_tx_branches, config.num_active_subcarriers),
            dtype=np.complex64,
        )
        for branch, centered_bin in enumerate(config.pilot_centered_bins):
            active_index = _centered_bin_to_active_index(
                centered_bin,
                config.num_active_subcarriers,
            )
            active[branch, active_index] = transmitted_pilots[
                repetition,
                branch,
            ]

        samples = ofdm_modulate_symbol(
            active,
            config.fft_size,
            config.cp_length_samples,
        )
        start = config.pilot_symbol_cp_start(repetition)
        stop = start + config.pilot_symbol_samples
        waveform[:, start:stop] = samples

    return np.ascontiguousarray(waveform, dtype=np.complex64)


def validate_sounding_period(
    waveform: NDArray[np.complexfloating],
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
    *,
    atol: float = 2e-6,
) -> None:
    """Raise if a generated period violates its timing or pilot allocation."""
    signal = np.asarray(waveform, dtype=np.complex64)
    expected_shape = (config.num_tx_branches, config.csi_interval_samples)
    if signal.shape != expected_shape:
        raise ValueError(f"expected waveform shape {expected_shape}, got {signal.shape}")
    if not np.all(np.isfinite(signal)):
        raise ValueError("waveform contains NaN or infinite samples")
    if np.max(np.abs(signal)) > config.digital_power_scale + atol:
        raise ValueError("waveform exceeds digital_power_scale")

    zc_stop = config.zc_length_samples
    guard_stop = zc_stop + config.guard_length_samples
    non_sync = np.arange(config.num_tx_branches) != config.sync_tx_branch
    if not np.allclose(signal[non_sync, :zc_stop], 0.0, atol=atol):
        raise ValueError("ZC must be present only on sync_tx_branch")
    if not np.allclose(signal[:, zc_stop:guard_stop], 0.0, atol=atol):
        raise ValueError("the ZC-to-pilot guard interval is not zero")
    if not np.allclose(
        signal[:, config.sounding_burst_samples :],
        0.0,
        atol=atol,
    ):
        raise ValueError("samples after the sounding burst are not zero")

    expected_pilots = build_transmitted_pilots(config)
    for repetition in range(config.pilot_repetitions):
        cp_start = config.pilot_symbol_cp_start(repetition)
        data_start = config.pilot_symbol_data_start(repetition)
        useful = signal[:, data_start : data_start + config.fft_size]
        cyclic_prefix = signal[:, cp_start:data_start]
        if not np.allclose(
            cyclic_prefix,
            useful[:, -config.cp_length_samples :],
            atol=atol,
        ):
            raise ValueError(
                f"cyclic prefix mismatch at repetition {repetition}"
            )
        bins = np.fft.fft(useful, axis=-1, norm="ortho")
        for branch, centered_bin in enumerate(config.pilot_centered_bins):
            fft_index = centered_bin % config.fft_size
            if not np.isclose(
                bins[branch, fft_index],
                expected_pilots[repetition, branch],
                atol=atol,
            ):
                raise ValueError(
                    f"pilot mismatch at repetition {repetition}, branch {branch}"
                )
            other_bins = np.delete(bins[branch], fft_index)
            if not np.allclose(other_bins, 0.0, atol=atol):
                raise ValueError(
                    f"unexpected occupied bin at repetition {repetition}, "
                    f"branch {branch}"
                )


def _import_uhd() -> Any:
    try:
        import uhd  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "UHD Python bindings are not installed. Install the UHD package "
            "matching the USRP host driver, or run with --dry-run."
        ) from error
    return uhd


def _make_tune_request(uhd: Any, frequency_hz: float) -> Any:
    """Support both current and older UHD Python binding names."""
    tune_request = getattr(uhd.types, "TuneRequest", None)
    if tune_request is not None:
        return tune_request(frequency_hz)
    return uhd.libpyuhd.types.tune_request(frequency_hz)


def _wait_for_external_reference_lock(
    usrp: Any,
    num_mboards: int,
    timeout_s: float,
) -> None:
    """Wait until every motherboard exposing ``ref_locked`` reports lock."""
    monitored = [
        mboard
        for mboard in range(num_mboards)
        if "ref_locked" in usrp.get_mboard_sensor_names(mboard)
    ]
    if not monitored:
        print("Warning: no motherboard exposes a ref_locked sensor.")
        return

    deadline = time.monotonic() + timeout_s
    pending = monitored
    while pending and time.monotonic() < deadline:
        pending = [
            mboard
            for mboard in monitored
            if not usrp.get_mboard_sensor("ref_locked", mboard).to_bool()
        ]
        if pending:
            time.sleep(0.1)
    if pending:
        raise RuntimeError(
            "OctoClock 10 MHz reference did not lock on motherboard(s): "
            + ", ".join(map(str, pending))
        )


def _synchronize_usrp_time(usrp: Any, uhd: Any, num_mboards: int) -> None:
    """Set every motherboard time to zero on one common external PPS edge."""
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

    if hasattr(usrp, "get_time_synchronized"):
        if not usrp.get_time_synchronized():
            raise RuntimeError("UHD reports that motherboard times are not synchronized")

    # Last-PPS values do not include host query latency.  Retry protects
    # against the unlikely case in which reads straddle the next PPS edge.
    for _ in range(3):
        last_pps = [
            usrp.get_time_last_pps(mboard).get_real_secs()
            for mboard in range(num_mboards)
        ]
        if max(last_pps) - min(last_pps) <= 1e-9:
            return
        time.sleep(0.05)
    raise RuntimeError(f"motherboard PPS epochs differ: {last_pps}")


def initialize_transmitter(
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
    hardware: TransmitterHardwareConfig = DEFAULT_HARDWARE_CONFIG,
    *,
    check_reference_lock: bool = True,
    reset_hardware_time: bool = True,
) -> Any:
    """Create and configure the OctoClock-synchronized three-X310 array."""
    hardware.validate(config)
    uhd = _import_uhd()
    device_args = hardware.device_args(config)
    print(f"Opening USRPs: {device_args}")
    usrp = uhd.usrp.MultiUSRP(device_args)

    num_mboards = usrp.get_num_mboards()
    if num_mboards != config.num_rus:
        raise RuntimeError(
            f"UHD discovered {num_mboards} motherboard(s); "
            f"the configuration requires {config.num_rus}"
        )

    for mboard in range(num_mboards):
        usrp.set_clock_source(config.clock_source, mboard)
        usrp.set_time_source(config.time_source, mboard)
        usrp.set_tx_subdev_spec(
            uhd.usrp.SubdevSpec(hardware.tx_subdev_spec),
            mboard,
        )

    if check_reference_lock:
        _wait_for_external_reference_lock(
            usrp,
            num_mboards,
            hardware.reference_lock_timeout_s,
        )

    for mboard in range(num_mboards):
        actual_master_clock = float(usrp.get_master_clock_rate(mboard))
        if not math.isclose(
            actual_master_clock,
            config.master_clock_rate_hz,
            rel_tol=1e-9,
            abs_tol=1.0,
        ):
            raise RuntimeError(
                f"motherboard {mboard} master clock is "
                f"{actual_master_clock} Hz, not {config.master_clock_rate_hz} Hz"
            )
    if reset_hardware_time:
        _synchronize_usrp_time(usrp, uhd, num_mboards)

    if usrp.get_tx_num_channels() != config.num_tx_branches:
        raise RuntimeError(
            f"UHD exposes {usrp.get_tx_num_channels()} TX channel(s); "
            f"the waveform requires {config.num_tx_branches}"
        )

    for channel in range(config.num_tx_branches):
        usrp.set_tx_rate(config.sampling_rate_hz, channel)
        usrp.set_tx_freq(
            _make_tune_request(uhd, config.center_frequency_hz),
            channel,
        )
        usrp.set_tx_gain(config.tx_gain_db, channel)
        usrp.set_tx_bandwidth(config.nominal_bandwidth_hz, channel)
        usrp.set_tx_antenna(hardware.tx_antenna, channel)

    for channel, label in enumerate(config.branch_labels):
        actual_rate = float(usrp.get_tx_rate(channel))
        if not math.isclose(
            actual_rate,
            config.sampling_rate_hz,
            rel_tol=1e-6,
            abs_tol=1.0,
        ):
            raise RuntimeError(
                f"TX channel {channel} rate is {actual_rate}, not the required "
                f"{config.sampling_rate_hz} S/s"
            )
        print(
            f"TX {channel}: {label}, pilot k={config.pilot_centered_bins[channel]:+d}, "
            f"rate={actual_rate:.0f} S/s, "
            f"frequency={usrp.get_tx_freq(channel):.0f} Hz, "
            f"gain={usrp.get_tx_gain(channel):.1f} dB"
        )
    return usrp


def choose_integer_second_start_time(
    usrp: Any,
    minimum_lead_time_s: float,
) -> float:
    """Choose a future integer UHD second shared by transmitter and receiver."""
    now = float(usrp.get_time_now().get_real_secs())
    # One extra integer second leaves time for streamer construction and lets
    # a separately launched receiver arm itself against the same UHD epoch.
    return float(math.ceil(now + minimum_lead_time_s) + 1)


def transmit_sounding(
    usrp: Any,
    waveform: ComplexArray,
    start_time_s: float,
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
    hardware: TransmitterHardwareConfig = DEFAULT_HARDWARE_CONFIG,
    *,
    num_periods: int | None,
    stop_event: Event | None = None,
) -> int:
    """Transmit periods continuously from one common timed UHD start.

    ``num_periods=None`` runs until interrupted.  The returned value is the
    number of complete 5 ms periods handed to UHD.
    """
    if num_periods is not None and num_periods <= 0:
        raise ValueError("num_periods must be positive or None")
    validate_sounding_period(waveform, config)

    uhd = _import_uhd()
    now = float(usrp.get_time_now().get_real_secs())
    if start_time_s - now < config.startup_lead_time_s:
        raise ValueError(
            f"start_time_s must be at least {config.startup_lead_time_s} s "
            f"in the future (now={now:.6f}, requested={start_time_s:.6f})"
        )

    stream_args = uhd.usrp.StreamArgs(
        config.cpu_sample_format,
        config.wire_sample_format,
    )
    stream_args.channels = list(range(config.num_tx_branches))
    streamer = usrp.get_tx_stream(stream_args)

    metadata = uhd.types.TXMetadata()
    metadata.start_of_burst = True
    metadata.end_of_burst = False
    metadata.has_time_spec = True
    metadata.time_spec = uhd.types.TimeSpec(start_time_s)

    periods_sent = 0
    first_send = True
    try:
        while (
            (num_periods is None or periods_sent < num_periods)
            and (stop_event is None or not stop_event.is_set())
        ):
            chunk_periods = hardware.periods_per_stream_chunk
            if num_periods is not None:
                chunk_periods = min(chunk_periods, num_periods - periods_sent)
            chunk = np.ascontiguousarray(
                np.tile(waveform, (1, chunk_periods)),
                dtype=np.complex64,
            )

            sample_offset = 0
            while sample_offset < chunk.shape[1]:
                sent = int(streamer.send(chunk[:, sample_offset:], metadata))
                if sent <= 0:
                    raise RuntimeError("UHD TX streamer accepted zero samples")
                sample_offset += sent
                if first_send:
                    first_send = False
                    metadata.start_of_burst = False
                    metadata.has_time_spec = False
            periods_sent += chunk_periods
    finally:
        end_metadata = uhd.types.TXMetadata()
        end_metadata.start_of_burst = False
        end_metadata.end_of_burst = True
        end_metadata.has_time_spec = False
        streamer.send(
            np.zeros((config.num_tx_branches, 0), dtype=np.complex64),
            end_metadata,
        )
    return periods_sent


def describe_waveform(
    waveform: ComplexArray,
    config: SoundingConfig = DEFAULT_SOUNDING_CONFIG,
) -> None:
    duration_ms = 1e3 * waveform.shape[1] / config.sampling_rate_hz
    print("Sounding waveform validated")
    print(f"  shape: {waveform.shape} (TX branches, time samples)")
    print(f"  sample rate: {config.sampling_rate_hz / 1e6:.6f} MS/s")
    print(f"  period: {waveform.shape[1]} samples = {duration_ms:.3f} ms")
    print(f"  ZC: samples 0..{config.zc_length_samples - 1}")
    print(
        "  guard: samples "
        f"{config.zc_length_samples}..{config.pilot_block_cp_start - 1}"
    )
    print(
        f"  pilots: {config.pilot_repetitions} x "
        f"{config.pilot_symbol_samples} samples, "
        f"starting at {config.pilot_block_cp_start}"
    )
    print(
        f"  zero padding: {config.zero_padding_samples} samples, "
        f"starting at {config.sounding_burst_samples}"
    )
    print(f"  peak magnitude: {np.max(np.abs(waveform)):.6f}")
    print(f"  CSI snapshots: {1.0 / config.csi_interval_s:.1f}/s")


def _parse_addresses(value: str) -> tuple[str, ...]:
    addresses = tuple(part.strip() for part in value.split(",") if part.strip())
    if not addresses:
        raise argparse.ArgumentTypeError("at least one IP address is required")
    return addresses


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--addresses",
        type=_parse_addresses,
        default=DEFAULT_USRP_ADDRESSES,
        help="comma-separated USRP IPs in RU order",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=200,
        help="number of 5 ms periods to send (default: 200 = 1 second)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="transmit until Ctrl-C instead of stopping after --periods",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="absolute UHD time in seconds; default is a future integer second",
    )
    parser.add_argument(
        "--chunk-periods",
        type=int,
        default=20,
        help="5 ms periods buffered per UHD send chunk",
    )
    parser.add_argument(
        "--skip-ref-lock-check",
        action="store_true",
        help="skip ref_locked sensor verification (not recommended)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="generate and validate the waveform without opening USRPs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = DEFAULT_SOUNDING_CONFIG
    hardware = TransmitterHardwareConfig(
        addresses=args.addresses,
        periods_per_stream_chunk=args.chunk_periods,
    )
    hardware.validate(config)

    waveform = build_sounding_period(config)
    validate_sounding_period(waveform, config)
    describe_waveform(waveform, config)
    if args.dry_run:
        print(f"  device args: {hardware.device_args(config)}")
        return 0

    usrp = initialize_transmitter(
        config,
        hardware,
        check_reference_lock=not args.skip_ref_lock_check,
    )
    start_time_s = (
        args.start_time
        if args.start_time is not None
        else choose_integer_second_start_time(usrp, config.startup_lead_time_s)
    )
    num_periods = None if args.continuous else args.periods
    print(
        f"Timed TX start: UHD {start_time_s:.6f} s; "
        + ("continuous" if num_periods is None else f"{num_periods} periods")
    )
    try:
        periods_sent = transmit_sounding(
            usrp,
            waveform,
            start_time_s,
            config,
            hardware,
            num_periods=num_periods,
        )
    except KeyboardInterrupt:
        print("Transmission stopped by user.")
        return 130
    print(
        f"Transmission complete: {periods_sent} periods "
        f"({periods_sent * config.csi_interval_s:.3f} s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HARDWARE_CONFIG",
    "DEFAULT_USRP_ADDRESSES",
    "TransmitterHardwareConfig",
    "build_sounding_period",
    "build_transmitted_pilots",
    "choose_integer_second_start_time",
    "initialize_transmitter",
    "transmit_sounding",
    "validate_sounding_period",
]
