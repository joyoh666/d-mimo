"""Timing and carrier-frequency synchronization for captured OFDM frames.

The equations and processing order are taken from
``revised_ofdm_channel_capture_virtual7_timeavg.py``.  The original file is
not modified; this module exposes the same operations as small reusable
functions driven by one shared :class:`config.OFDMConfig` instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate

try:  # Support script execution and package-style imports.
    from .config import DEFAULT_CONFIG, OFDMConfig
except ImportError:
    from USRP.revised_ofdm.config import DEFAULT_CONFIG, OFDMConfig


ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SynchronizationResult:
    """Intermediate and final outputs from the complete synchronization chain."""

    frame_rcv: ComplexArray
    frame_rcv_timesync: ComplexArray
    frame_rcv_ffosync: ComplexArray
    frame_rcv_rfosync: ComplexArray
    sync_idx: int
    ffo_hz: float
    rfo_hz_per_slot: FloatArray
    timing_correlation_peak: float
    timing_sync_fallback_used: bool

    @property
    def frame_rcv_ifosync(self) -> ComplexArray:
        """Integer-FO synchronization is disabled, as in the original code."""
        return self.frame_rcv

    def to_legacy_dict(self) -> dict[str, object]:
        """Return names that can be merged into the original result dictionary."""
        return {
            "frame_rcv": self.frame_rcv,
            "frame_rcv_timesync": self.frame_rcv_timesync,
            "frame_rcv_ffosync": self.frame_rcv_ffosync,
            "frame_rcv_rfosync": self.frame_rcv_rfosync,
            "sync_idx": self.sync_idx,
            "ffo_hz": self.ffo_hz,
            "rfo_hz_per_slot": self.rfo_hz_per_slot,
            "timing_correlation_peak": self.timing_correlation_peak,
            "timing_sync_fallback_used": self.timing_sync_fallback_used,
        }


def preprocess_received_signal(
    frame_rcv: NDArray[np.complexfloating],
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    receive_row: int = 0,
) -> ComplexArray:
    """Discard initial samples, remove DC, and select one RX stream.

    ``receive_row`` is the row within the array returned by ``sendAndReceive``;
    it is not the UHD hardware channel number.  A one-dimensional input is also
    accepted for saved-IQ reprocessing.
    """
    samples = np.asarray(frame_rcv)
    if samples.ndim not in (1, 2):
        raise ValueError(
            f"frame_rcv must be one- or two-dimensional, got shape {samples.shape}"
        )
    if samples.shape[-1] <= config.RX_DISCARD_SAMPLES:
        raise ValueError("frame_rcv is shorter than RX_DISCARD_SAMPLES")

    if samples.ndim == 1:
        if receive_row != 0:
            raise ValueError("receive_row must be 0 for one-dimensional input")
        prepared = samples[config.RX_DISCARD_SAMPLES:].copy()
    else:
        if not 0 <= receive_row < samples.shape[0]:
            raise ValueError(
                f"receive_row={receive_row} is outside {samples.shape[0]} RX rows"
            )
        prepared = samples[:, config.RX_DISCARD_SAMPLES:].copy()

    # This intentionally uses one global mean, matching the original code.
    prepared -= np.mean(prepared)
    if prepared.ndim == 2:
        prepared = prepared[receive_row]
    return np.asarray(prepared, dtype=np.complex64).flatten()


def timing_synchronize(
    frame_rcv_ifosync: ComplexArray,
    ss_td_with_cp: ComplexArray,
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    report_fallback: bool = True,
) -> tuple[ComplexArray, int, float, bool]:
    """Locate the frame using FFT correlation with the sync template."""
    frame = np.asarray(frame_rcv_ifosync, dtype=np.complex64).flatten()
    sync_template = np.asarray(ss_td_with_cp, dtype=np.complex64).flatten()
    if sync_template.size == 0:
        raise ValueError("ss_td_with_cp must not be empty")
    if frame.size < sync_template.size:
        raise ValueError("Received signal is shorter than the sync template")

    correlation = np.abs(correlate(frame, sync_template, "valid", "fft"))
    ss_start_idx = (
        config.FFT_SIZE * 5
        + config.first_CP_length
        + config.normal_CP_length * 4
    )
    sync_idx = int(np.argmax(correlation) - ss_start_idx)

    fallback_used = sync_idx < 0 or sync_idx + config.frame_length > frame.size
    if fallback_used:
        if report_fallback:
            print("sync not found, using sync_idx = 0")
        sync_idx = 0

    synchronized = frame[sync_idx:sync_idx + config.frame_length]
    if synchronized.size < config.frame_length:
        synchronized = np.concatenate(
            [
                synchronized.astype(np.complex64),
                np.zeros(
                    config.frame_length - synchronized.size,
                    dtype=np.complex64,
                ),
            ]
        )

    correlation_peak = float(np.max(correlation)) if correlation.size else 0.0
    return synchronized.astype(np.complex64), sync_idx, correlation_peak, fallback_used


def fractional_frequency_synchronize(
    frame_rcv_timesync: ComplexArray,
    config: OFDMConfig = DEFAULT_CONFIG,
) -> tuple[ComplexArray, float]:
    """Estimate and compensate fractional frequency offset using all CPs."""
    frame = np.asarray(frame_rcv_timesync, dtype=np.complex64).flatten()
    if frame.size != config.frame_length:
        raise ValueError(
            f"Time-synchronized frame must contain {config.frame_length} samples, "
            f"got {frame.size}"
        )

    num_slots = config.num_subframe_per_frame * config.num_slot_per_subframe
    td_symbols_rcv = np.reshape(frame, (num_slots, -1))

    first_CPs = td_symbols_rcv[:, :config.first_CP_length]
    first_signal_for_CPs = td_symbols_rcv[
        :,
        config.FFT_SIZE:config.FFT_SIZE + config.first_CP_length,
    ]

    td_symbols_rcv_wo_first = td_symbols_rcv[
        :,
        config.FFT_SIZE + config.first_CP_length:,
    ]
    td_symbols_rcv_wo_first = np.reshape(
        td_symbols_rcv_wo_first,
        (num_slots * (config.num_symbols_per_slot - 1), -1),
    )
    other_CPs = td_symbols_rcv_wo_first[:, :config.normal_CP_length]
    other_signal_for_CPs = td_symbols_rcv_wo_first[
        :,
        -config.normal_CP_length:,
    ]

    phase_diff = np.angle(
        np.sum(np.conj(other_CPs) * other_signal_for_CPs)
        + np.sum(np.conj(first_CPs) * first_signal_for_CPs)
    )
    ffo_hz = float(
        phase_diff * config.sampling_rate / (2 * np.pi * config.FFT_SIZE)
    )

    sample_index = np.arange(frame.size)
    compensation = np.exp(
        -1j
        * ffo_hz
        * sample_index
        / config.sampling_rate
        * 2
        * np.pi
    ).astype(np.complex64)
    corrected = frame * compensation
    return corrected.astype(np.complex64), ffo_hz


def residual_frequency_synchronize(
    frame_rcv_ffosync: ComplexArray,
    config: OFDMConfig = DEFAULT_CONFIG,
) -> tuple[ComplexArray, FloatArray]:
    """Estimate and compensate slot-wise residual FO from symbol-0 references."""
    frame = np.asarray(frame_rcv_ffosync, dtype=np.complex64).flatten()
    if frame.size != config.frame_length:
        raise ValueError(
            f"FFO-corrected frame must contain {config.frame_length} samples, "
            f"got {frame.size}"
        )

    num_slots = config.num_subframe_per_frame * config.num_slot_per_subframe
    if num_slots < 2:
        raise ValueError("Residual FO estimation requires at least two slots")

    refsym_rcv_td = np.reshape(frame, (num_slots, -1))
    refsym_rcv_td = refsym_rcv_td[
        :,
        config.first_CP_length:config.first_CP_length + config.FFT_SIZE,
    ]
    phase_diff = np.angle(
        np.sum(
            np.conj(refsym_rcv_td[:-1]) * refsym_rcv_td[1:],
            axis=-1,
        )
    )
    rfo_hz = (
        phase_diff
        * config.sampling_rate
        / (2 * np.pi * config.slot_length)
    )
    rfo_hz = np.concatenate([rfo_hz, rfo_hz[-1:]], axis=0)
    rfo_column = np.expand_dims(rfo_hz, axis=1)

    slot_sample_index = np.expand_dims(np.arange(config.slot_length), axis=0)
    compensation = np.exp(
        -1j
        * rfo_column
        * slot_sample_index
        / config.sampling_rate
        * 2
        * np.pi
    ).astype(np.complex64)
    corrected = frame * compensation.flatten()
    return corrected.astype(np.complex64), rfo_hz.astype(np.float32)


def synchronize_received_frame(
    frame_rcv: NDArray[np.complexfloating],
    ss_td_with_cp: ComplexArray,
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    receive_row: int = 0,
    report_timing_fallback: bool = True,
) -> SynchronizationResult:
    """Run preprocessing, timing sync, FFO correction, and RFO correction."""
    preprocessed = preprocess_received_signal(
        frame_rcv,
        config,
        receive_row=receive_row,
    )
    time_synchronized, sync_idx, correlation_peak, fallback_used = (
        timing_synchronize(
            preprocessed,
            ss_td_with_cp,
            config,
            report_fallback=report_timing_fallback,
        )
    )
    ffo_corrected, ffo_hz = fractional_frequency_synchronize(
        time_synchronized,
        config,
    )
    rfo_corrected, rfo_hz_per_slot = residual_frequency_synchronize(
        ffo_corrected,
        config,
    )

    return SynchronizationResult(
        frame_rcv=preprocessed,
        frame_rcv_timesync=time_synchronized,
        frame_rcv_ffosync=ffo_corrected,
        frame_rcv_rfosync=rfo_corrected,
        sync_idx=sync_idx,
        ffo_hz=ffo_hz,
        rfo_hz_per_slot=rfo_hz_per_slot,
        timing_correlation_peak=correlation_peak,
        timing_sync_fallback_used=fallback_used,
    )
