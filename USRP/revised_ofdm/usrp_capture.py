"""USRP initialization and timed OFDM transmit/receive operations.

This module is the hardware boundary of the capture pipeline.  It deliberately
contains no synchronization, FFT, or channel-estimation logic.  Low-level UHD
streaming remains in ``usrp_utils.sendAndReceive`` and is reused unchanged.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import uhd

try:  # Support script execution and package-style imports.
    from .config import DEFAULT_CONFIG, OFDMConfig
    from ...usrp_utils import sendAndReceive
except ImportError:
    from USRP.test.revised_ofdm.config import DEFAULT_CONFIG, OFDMConfig
    from usrp_utils import sendAndReceive


ComplexArray = NDArray[np.complex64]


def initialize_usrp(
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    mboard: int = 0,
    clock_source: str | None = None,
    time_source: str | None = None,
    reset_time_at_unknown_pps: bool = True,
    verbose: bool = True,
) -> Any:
    """Create and configure the UHD MultiUSRP used by the original script.

    ``clock_source`` and ``time_source`` are optional because the revised
    original script does not set them.  Multi-USRP/RU experiments can pass
    ``"external"`` when an OctoClock or equivalent reference is connected.
    """
    usrp = uhd.usrp.MultiUSRP(config.USRP_DEVICE_ARGS)
    if not 0 <= mboard < usrp.get_num_mboards():
        raise ValueError(
            f"mboard={mboard} is outside {usrp.get_num_mboards()} motherboard(s)"
        )

    if clock_source is not None:
        usrp.set_clock_source(clock_source, mboard)
    if time_source is not None:
        usrp.set_time_source(time_source, mboard)
    if reset_time_at_unknown_pps:
        usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

    usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(config.TX_SUBDEV_SPEC), mboard)
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(config.RX_SUBDEV_SPEC), mboard)

    _validate_hardware_channels(usrp, config.TX_CHANNELS, config.RX_CHANNELS)
    for channel in config.TX_CHANNELS:
        usrp.set_tx_antenna(config.TX_ANTENNA, channel)
    for channel in config.RX_CHANNELS:
        usrp.set_rx_antenna(config.RX_ANTENNA, channel)

    if verbose:
        print("USRP loaded. Session Ready.")
        print(f"Tx subdevice: {usrp.get_tx_subdev_spec(mboard)}")
        print(f"Rx subdevice: {usrp.get_rx_subdev_spec(mboard)}")
        print(f"Tx channels: {config.TX_CHANNELS}")
        print(f"Rx channels: {config.RX_CHANNELS}")
    return usrp


def _validate_hardware_channels(
    usrp: Any,
    tx_channels: Sequence[int],
    rx_channels: Sequence[int],
) -> None:
    if not tx_channels:
        raise ValueError("At least one TX channel is required")
    if not rx_channels:
        raise ValueError("At least one RX channel is required")
    if any(channel < 0 for channel in (*tx_channels, *rx_channels)):
        raise ValueError("USRP channel indices must be non-negative")
    if max(tx_channels) >= usrp.get_tx_num_channels():
        raise ValueError(
            f"TX channel {max(tx_channels)} requested, but UHD exposes only "
            f"{usrp.get_tx_num_channels()} channel(s)"
        )
    if max(rx_channels) >= usrp.get_rx_num_channels():
        raise ValueError(
            f"RX channel {max(rx_channels)} requested, but UHD exposes only "
            f"{usrp.get_rx_num_channels()} channel(s)"
        )


def transmit_and_receive_ofdm(
    usrp: Any,
    waveform: NDArray[np.complexfloating],
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    tx_channels: Sequence[int] | None = None,
    rx_channels: Sequence[int] | None = None,
    stream_power: float = 1.0,
) -> ComplexArray:
    """Transmit one OFDM waveform and return raw samples from the USRP.

    The returned array has shape ``(num_rx_channels, num_received_samples)``.
    It is intentionally not trimmed or synchronized; that is handled by
    ``receiver_processing.process_received_frame``.
    """
    tx_channels = tuple(config.TX_CHANNELS if tx_channels is None else tx_channels)
    rx_channels = tuple(config.RX_CHANNELS if rx_channels is None else rx_channels)
    _validate_hardware_channels(usrp, tx_channels, rx_channels)

    signal = np.asarray(waveform, dtype=np.complex64)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.ndim != 2:
        raise ValueError(f"waveform must be 1D or 2D, got shape {signal.shape}")
    if signal.shape[0] != len(tx_channels):
        raise ValueError(
            f"waveform has {signal.shape[0]} TX row(s), but "
            f"{len(tx_channels)} TX channel(s) were selected"
        )
    if signal.shape[1] != config.frame_length:
        raise ValueError(
            f"waveform must contain {config.frame_length} samples per TX row, "
            f"got {signal.shape[1]}"
        )
    if stream_power <= 0:
        raise ValueError("stream_power must be positive")

    tx_delay_samples = int(config.sampling_rate * config.TX_DELAY_TIME_S)
    rx_trailing_samples = int(config.sampling_rate * config.RX_TRAILING_TIME_S)
    received = sendAndReceive(
        usrp,
        signal.copy(),
        stream_power,
        config.carrier_frequency,
        config.sampling_rate,
        config.Tx_gain,
        config.Rx_gain,
        list(tx_channels),
        list(rx_channels),
        wait_time=config.WAIT_TIME_S,
        tx_delay_samples=tx_delay_samples,
        rx_trailing_samples=rx_trailing_samples,
        otw_format=config.OTW_FORMAT,
    )
    return np.asarray(received, dtype=np.complex64)
