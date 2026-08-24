"""Transmit the six-branch D-MIMO sounding frame with UHD.

The default mode is a hardware-free dry run. RF transmission starts only when
``--transmit`` is explicitly supplied. For three two-channel USRPs, group the
devices into one MultiUSRP with addr0/addr1/addr2 and connect every motherboard
to the same external 10 MHz and PPS references.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from config import SoundingConfig
from transmitter import build_transmit_frame


Complex64Array = NDArray[np.complex64]


@dataclass(frozen=True)
class TxRuntimeConfig:
    device_args: str
    channels: tuple[int, ...]
    center_frequency_hz: float
    bandwidth_hz: float
    gain_db: float
    antenna: str | None
    clock_source: str
    time_source: str
    synchronize_time: bool
    start_delay_s: float
    frames: int
    amplitude: float
    ref_lock_timeout_s: float


def parse_channel_list(text: str) -> tuple[int, ...]:
    """Parse a comma-separated UHD channel list."""
    try:
        channels = tuple(int(item.strip()) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Channels must be comma-separated integers."
        ) from exc
    if not channels or any(channel < 0 for channel in channels):
        raise argparse.ArgumentTypeError("Channels must be non-negative.")
    if len(set(channels)) != len(channels):
        raise argparse.ArgumentTypeError("Channels must be unique.")
    return channels


def prepare_iq_buffer(cfg: SoundingConfig, amplitude: float) -> Complex64Array:
    """Build and peak-scale the six-channel waveform for fc32 streaming."""
    if not 0.0 < amplitude <= 1.0:
        raise ValueError("amplitude must be in the interval (0, 1].")

    frame = build_transmit_frame(cfg).branch_samples
    peak = float(np.max(np.abs(frame)))
    if peak == 0.0:
        raise ValueError("The generated transmit frame is empty.")

    scaled = frame * (amplitude / peak)
    return np.ascontiguousarray(scaled, dtype=np.complex64)


def wait_for_reference_lock(usrp: Any, timeout_s: float) -> None:
    """Wait until every motherboard that exposes ref_locked is locked."""
    deadline = time.monotonic() + timeout_s
    pending: set[int] = set()

    for motherboard in range(usrp.get_num_mboards()):
        names = set(usrp.get_mboard_sensor_names(motherboard))
        if "ref_locked" in names:
            pending.add(motherboard)

    if not pending:
        print("No ref_locked sensor is exposed; skipping lock polling.")
        return

    while pending and time.monotonic() < deadline:
        for motherboard in tuple(pending):
            if usrp.get_mboard_sensor("ref_locked", motherboard).to_bool():
                pending.remove(motherboard)
        if pending:
            time.sleep(0.1)

    if pending:
        boards = ", ".join(str(index) for index in sorted(pending))
        raise RuntimeError(
            f"External reference did not lock on motherboard(s): {boards}."
        )


def configure_references(
    usrp: Any,
    uhd: Any,
    runtime: TxRuntimeConfig,
) -> None:
    """Select common references and synchronize all motherboard times."""
    for motherboard in range(usrp.get_num_mboards()):
        usrp.set_clock_source(runtime.clock_source, motherboard)
        usrp.set_time_source(runtime.time_source, motherboard)

    if runtime.clock_source != "internal":
        wait_for_reference_lock(usrp, runtime.ref_lock_timeout_s)

    if runtime.synchronize_time:
        print("Synchronizing motherboard times at the next PPS edge...")
        usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))


def configure_frontends(
    usrp: Any,
    sounding: SoundingConfig,
    runtime: TxRuntimeConfig,
) -> None:
    """Validate channel mapping and configure RF bandwidth/antenna ports."""
    available = int(usrp.get_tx_num_channels())
    if any(channel >= available for channel in runtime.channels):
        raise RuntimeError(
            f"Requested {runtime.channels}, but UHD exposes only "
            f"{available} TX channel(s)."
        )

    for channel in runtime.channels:
        usrp.set_tx_rate(sounding.sample_rate_hz, channel)
        usrp.set_tx_gain(runtime.gain_db, channel)
        usrp.set_tx_bandwidth(runtime.bandwidth_hz, channel)
        if runtime.antenna:
            usrp.set_tx_antenna(runtime.antenna, channel)


def validate_achieved_rates(
    usrp: Any,
    sounding: SoundingConfig,
    runtime: TxRuntimeConfig,
) -> None:
    """Reject a coerced sample rate that would change the OFDM SCS."""
    for channel in runtime.channels:
        actual_rate = float(usrp.get_tx_rate(channel))
        error_ppm = (
            abs(actual_rate - sounding.sample_rate_hz)
            / sounding.sample_rate_hz
            * 1e6
        )
        if error_ppm > 1.0:
            raise RuntimeError(
                f"Channel {channel} rate was coerced to {actual_rate:.6f} S/s "
                f"({error_ppm:.2f} ppm error). Choose a compatible master "
                "clock rate in --device-args."
            )


def print_achieved_settings(
    usrp: Any,
    sounding: SoundingConfig,
    runtime: TxRuntimeConfig,
) -> None:
    """Print the settings reported by UHD after transmission."""
    validate_achieved_rates(usrp, sounding, runtime)
    for channel in runtime.channels:
        actual_rate = float(usrp.get_tx_rate(channel))

        print(
            f"TX channel {channel}: rate={actual_rate / 1e6:.6f} MS/s, "
            f"freq={usrp.get_tx_freq(channel) / 1e9:.9f} GHz, "
            f"gain={usrp.get_tx_gain(channel):.1f} dB"
        )


def print_plan(
    sounding: SoundingConfig,
    runtime: TxRuntimeConfig,
    frame: Complex64Array,
) -> None:
    """Print the exact waveform and channel mapping before transmission."""
    print("=== D-MIMO UHD transmit plan ===")
    print(f"Device args       : {runtime.device_args or '(not set)'}")
    print(f"Branch→channels   : {runtime.channels}")
    print(f"IQ buffer shape   : {frame.shape}")
    print(f"IQ dtype          : {frame.dtype}")
    print(f"Peak magnitude    : {np.max(np.abs(frame)):.4f}")
    print(f"Center frequency  : {runtime.center_frequency_hz / 1e9:.6f} GHz")
    print(f"Sample rate       : {sounding.sample_rate_hz / 1e6:.6f} MS/s")
    print(f"Frame period      : {sounding.pilot_period_s * 1e3:.3f} ms")
    print(f"Frames            : {runtime.frames}")
    print(f"TX duration       : {runtime.frames * sounding.pilot_period_s:.6f} s")
    print(f"Clock/time source : {runtime.clock_source}/{runtime.time_source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transmit the six-branch D-MIMO sounding waveform. Without "
            "--transmit, only waveform validation is performed."
        )
    )
    parser.add_argument(
        "--device-args",
        default="",
        help=(
            "UHD arguments, e.g. addr0=192.168.10.2,"
            "addr1=192.168.20.2,addr2=192.168.30.2"
        ),
    )
    parser.add_argument(
        "--channels",
        type=parse_channel_list,
        default=parse_channel_list("0,1,2,3,4,5"),
        help="Comma-separated UHD TX channels in branch order.",
    )
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--tx-gain-db", type=float, default=0.0)
    parser.add_argument("--bandwidth-hz", type=float, default=1.4e6)
    parser.add_argument("--center-frequency-hz", type=float, default=2.2e9)
    parser.add_argument("--antenna", default="TX/RX")
    parser.add_argument("--clock-source", default="external")
    parser.add_argument("--time-source", default="external")
    parser.add_argument(
        "--synchronize-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize motherboard device times on a PPS edge.",
    )
    parser.add_argument("--start-delay-s", type=float, default=1.0)
    parser.add_argument("--ref-lock-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--transmit",
        action="store_true",
        help="Actually open UHD and transmit RF. Omit for a dry run.",
    )
    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    sounding: SoundingConfig,
) -> TxRuntimeConfig:
    if len(args.channels) != sounding.num_tx_branches:
        parser.error(
            f"Exactly {sounding.num_tx_branches} UHD channels are required."
        )
    if args.frames <= 0:
        parser.error("--frames must be positive.")
    if not 0.0 < args.amplitude <= 1.0:
        parser.error("--amplitude must be in the interval (0, 1].")
    if args.start_delay_s <= 0.0:
        parser.error("--start-delay-s must be positive for timed TX.")
    if args.ref_lock_timeout_s <= 0.0:
        parser.error("--ref-lock-timeout-s must be positive.")
    if args.transmit and not args.device_args:
        parser.error(
            "--device-args is required with --transmit to prevent opening "
            "an unintended USRP."
        )

    return TxRuntimeConfig(
        device_args=args.device_args,
        channels=args.channels,
        center_frequency_hz=args.center_frequency_hz,
        bandwidth_hz=args.bandwidth_hz,
        gain_db=args.tx_gain_db,
        antenna=args.antenna.strip() or None,
        clock_source=args.clock_source,
        time_source=args.time_source,
        synchronize_time=args.synchronize_time,
        start_delay_s=args.start_delay_s,
        frames=args.frames,
        amplitude=args.amplitude,
        ref_lock_timeout_s=args.ref_lock_timeout_s,
    )


def import_uhd() -> Any:
    try:
        import uhd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The UHD Python module is unavailable in this interpreter. Use "
            "the Python installation that contains pyuhd."
        ) from exc
    return uhd


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sounding = SoundingConfig()
    sounding.validate()
    runtime = validate_arguments(parser, args, sounding)
    frame = prepare_iq_buffer(sounding, runtime.amplitude)
    print_plan(sounding, runtime, frame)

    if not args.transmit:
        print("Dry run complete. Use --transmit only in an RF-safe test setup.")
        return 0

    uhd = import_uhd()
    print("Opening UHD MultiUSRP...")
    usrp = uhd.usrp.MultiUSRP(runtime.device_args)
    
    configure_references(usrp, uhd, runtime)
    configure_frontends(usrp, sounding, runtime)
    validate_achieved_rates(usrp, sounding, runtime)

    duration_s = runtime.frames * sounding.pilot_period_s
    start_time = usrp.get_time_now() + runtime.start_delay_s
    sent = int(
        usrp.send_waveform(
            frame,
            duration_s,

            runtime.center_frequency_hz,
            sounding.sample_rate_hz,
            list(runtime.channels),
            runtime.gain_db,
            start_time,
        )
    )
    print_achieved_settings(usrp, sounding, runtime)

    expected = runtime.frames * sounding.period_samples
    if sent != expected:
        raise RuntimeError(f"Expected {expected} samples/channel, sent {sent}.")
    print(f"Transmission complete: {sent} samples per channel.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Transmission interrupted.", file=sys.stderr)
        raise SystemExit(130)
