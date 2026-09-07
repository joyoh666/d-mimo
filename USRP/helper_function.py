"""Reusable DSP and timing helpers for the USRP sounding pipeline.

The functions in this module are deliberately independent of UHD, ROS 2, and
the experiment-specific ``OFDMConfig`` class.  They can therefore be shared by
the paper-compatible six-branch sounder, offline tests, and later DFT beam
sweeping code.

Active-subcarrier arrays use the ordering already used in this repository:
negative-frequency subcarriers first and positive-frequency subcarriers
second.  DC is not included in the active-subcarrier array.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import correlate


ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float64]


def get_zc_sequence(length: int, root: int = 25) -> ComplexArray:
    """Return the Zadoff-Chu sequence used by the existing sounding code."""
    if length <= 0:
        raise ValueError("length must be positive")
    if root <= 0:
        raise ValueError("root must be positive")
    if math.gcd(length, root) != 1:
        raise ValueError("length and root must be coprime")

    sample_index = np.arange(length, dtype=np.float64)
    phase = -np.pi * root * sample_index * (sample_index + 1.0) / length
    return np.exp(1j * phase).astype(np.complex64)


def active_to_fft_bins(
    active_subcarriers: ArrayLike,
    fft_size: int,
) -> ComplexArray:
    """Map active subcarriers to FFT bins while leaving the DC bin unused.

    The final axis is interpreted as active subcarriers.  Leading dimensions
    such as branch, frame, or OFDM-symbol axes are preserved.
    """
    active = np.asarray(active_subcarriers, dtype=np.complex64)
    if active.ndim == 0:
        raise ValueError("active_subcarriers must have at least one dimension")
    num_active = active.shape[-1]
    if num_active <= 0 or num_active % 2:
        raise ValueError(
            "the number of active subcarriers must be positive and even"
        )
    if fft_size <= num_active:
        raise ValueError(
            "fft_size must be greater than the number of active subcarriers"
        )

    fft_bins = np.zeros((*active.shape[:-1], fft_size), dtype=np.complex64)
    half = num_active // 2
    fft_bins[..., 1 : half + 1] = active[..., half:]
    fft_bins[..., -half:] = active[..., :half]
    return fft_bins


def fft_bins_to_active(
    fft_bins: ArrayLike,
    num_active_subcarriers: int,
) -> ComplexArray:
    """Restore repository-order active subcarriers from an FFT-bin array."""
    bins = np.asarray(fft_bins, dtype=np.complex64)
    if bins.ndim == 0:
        raise ValueError("fft_bins must have at least one dimension")
    if num_active_subcarriers <= 0 or num_active_subcarriers % 2:
        raise ValueError("num_active_subcarriers must be positive and even")
    if bins.shape[-1] <= num_active_subcarriers:
        raise ValueError("FFT-bin count must exceed num_active_subcarriers")

    half = num_active_subcarriers // 2
    active = np.empty(
        (*bins.shape[:-1], num_active_subcarriers),
        dtype=np.complex64,
    )
    active[..., :half] = bins[..., -half:]
    active[..., half:] = bins[..., 1 : half + 1]
    return active


def ofdm_modulate_symbol(
    active_subcarriers: ArrayLike,
    fft_size: int,
    cp_length: int,
) -> ComplexArray:
    """Map, IFFT, and prepend a cyclic prefix to one or more OFDM symbols."""
    if cp_length < 0 or cp_length > fft_size:
        raise ValueError("cp_length must satisfy 0 <= cp_length <= fft_size")

    fft_bins = active_to_fft_bins(active_subcarriers, fft_size)
    useful = np.fft.ifft(fft_bins, axis=-1, norm="ortho").astype(
        np.complex64
    )
    if cp_length == 0:
        return useful
    return np.concatenate((useful[..., -cp_length:], useful), axis=-1)


def ofdm_demodulate_symbol(
    samples_with_cp: ArrayLike,
    fft_size: int,
    cp_length: int,
    num_active_subcarriers: int,
) -> ComplexArray:
    """Remove a cyclic prefix, FFT, and return the active subcarriers."""
    samples = np.asarray(samples_with_cp, dtype=np.complex64)
    expected_length = fft_size + cp_length
    if samples.ndim == 0 or samples.shape[-1] != expected_length:
        raise ValueError(
            f"the final sample axis must have length {expected_length}"
        )

    useful = samples[..., cp_length:]
    fft_bins = np.fft.fft(useful, axis=-1, norm="ortho").astype(np.complex64)
    return fft_bins_to_active(fft_bins, num_active_subcarriers)


def find_sequence_start(
    received_samples: ArrayLike,
    reference_sequence: ArrayLike,
    *,
    reference_offset_in_frame: int = 0,
) -> tuple[int, float]:
    """Locate a known sequence and return ``(frame_start, peak_magnitude)``.

    ``reference_offset_in_frame`` is the reference sequence's sample offset
    from the desired frame start.  Supplying it avoids waveform-layout
    constants inside the synchronization algorithm.
    """
    received = np.asarray(received_samples, dtype=np.complex64).reshape(-1)
    reference = np.asarray(reference_sequence, dtype=np.complex64).reshape(-1)
    if reference.size == 0:
        raise ValueError("reference_sequence must not be empty")
    if received.size < reference.size:
        raise ValueError("received_samples is shorter than reference_sequence")
    if reference_offset_in_frame < 0:
        raise ValueError("reference_offset_in_frame must be non-negative")

    correlation = np.abs(
        correlate(received, reference, mode="valid", method="fft")
    )
    reference_start = int(np.argmax(correlation))
    frame_start = reference_start - reference_offset_in_frame
    peak = float(correlation[reference_start])
    return frame_start, peak


def estimate_cfo_from_cp(
    samples_with_cp: ArrayLike,
    fft_size: int,
    cp_length: int,
    sampling_rate_hz: float,
) -> float:
    """Estimate fractional carrier-frequency offset from a cyclic prefix."""
    samples = np.asarray(samples_with_cp, dtype=np.complex64).reshape(-1)
    expected_length = fft_size + cp_length
    if cp_length <= 0 or cp_length > fft_size:
        raise ValueError("cp_length must satisfy 0 < cp_length <= fft_size")
    if samples.size != expected_length:
        raise ValueError(
            f"samples_with_cp must contain {expected_length} samples"
        )
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    cyclic_prefix = samples[:cp_length]
    repeated_tail = samples[fft_size : fft_size + cp_length]
    phase = np.angle(np.vdot(cyclic_prefix, repeated_tail))
    return float(phase * sampling_rate_hz / (2.0 * np.pi * fft_size))


def compensate_cfo(
    samples: ArrayLike,
    cfo_hz: float,
    sampling_rate_hz: float,
    *,
    first_sample_index: int = 0,
) -> ComplexArray:
    """Compensate a carrier-frequency offset along the final sample axis."""
    signal = np.asarray(samples, dtype=np.complex64)
    if signal.ndim == 0:
        raise ValueError("samples must have at least one dimension")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if first_sample_index < 0:
        raise ValueError("first_sample_index must be non-negative")

    indices = first_sample_index + np.arange(
        signal.shape[-1],
        dtype=np.float64,
    )
    correction = np.exp(
        -1j * 2.0 * np.pi * cfo_hz * indices / sampling_rate_hz
    ).astype(np.complex64)
    return (signal * correction).astype(np.complex64)


def least_squares_channel_estimate(
    received_pilots: ArrayLike,
    transmitted_pilots: ArrayLike,
) -> ComplexArray:
    """Return the element-wise LS channel estimate ``H = Y / X``."""
    received = np.asarray(received_pilots, dtype=np.complex64)
    transmitted = np.asarray(transmitted_pilots, dtype=np.complex64)
    try:
        received, transmitted = np.broadcast_arrays(received, transmitted)
    except ValueError as error:
        raise ValueError(
            "received and transmitted pilots are not broadcastable"
        ) from error
    if np.any(np.abs(transmitted) == 0):
        raise ValueError("transmitted_pilots must not contain zero-valued pilots")
    return (received / transmitted).astype(np.complex64)


def phase_align_channel_estimates(
    channel_estimates: ArrayLike,
) -> tuple[ComplexArray, NDArray[np.float32]]:
    """Align repeated CSI vectors to the common phase of the first vector."""
    estimates = np.asarray(channel_estimates, dtype=np.complex64)
    if estimates.ndim != 2 or estimates.shape[0] == 0:
        raise ValueError(
            "channel_estimates must have shape (observations, subcarriers)"
        )

    reference = estimates[0]
    aligned = np.empty_like(estimates)
    phases = np.zeros(estimates.shape[0], dtype=np.float32)
    aligned[0] = reference
    for index in range(1, estimates.shape[0]):
        phase = float(np.angle(np.vdot(reference, estimates[index])))
        phases[index] = phase
        aligned[index] = estimates[index] * np.exp(-1j * phase)
    return aligned.astype(np.complex64), phases


def active_channel_to_delay_response(
    active_channel: ArrayLike,
    fft_size: int,
) -> tuple[ComplexArray, ComplexArray]:
    """Return ``(CIR, full_spectrum)`` for active-subcarrier CSI vectors."""
    full_spectrum = active_to_fft_bins(active_channel, fft_size)
    impulse_response = np.fft.ifft(
        full_spectrum,
        axis=-1,
        norm="ortho",
    ).astype(np.complex64)
    return impulse_response, full_spectrum


def sample_timestamps(
    first_sample_time_s: float,
    sample_offsets: ArrayLike,
    sampling_rate_hz: float,
) -> FloatArray:
    """Convert RX-buffer sample offsets to USRP hardware timestamps."""
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    offsets = np.asarray(sample_offsets)
    if not np.issubdtype(offsets.dtype, np.integer):
        if np.any(offsets != np.floor(offsets)):
            raise ValueError("sample_offsets must contain integer sample indices")
    if np.any(offsets < 0):
        raise ValueError("sample_offsets must be non-negative")
    return first_sample_time_s + offsets.astype(np.float64) / sampling_rate_hz


def periodic_sample_offsets(
    interval_s: float,
    sampling_rate_hz: float,
    count: int,
    *,
    first_offset: int = 0,
) -> NDArray[np.int64]:
    """Return exact sample offsets for periodic events such as 5 ms pilots."""
    if interval_s <= 0 or sampling_rate_hz <= 0:
        raise ValueError("interval_s and sampling_rate_hz must be positive")
    if count < 0 or first_offset < 0:
        raise ValueError("count and first_offset must be non-negative")

    interval_samples_float = interval_s * sampling_rate_hz
    interval_samples = int(round(interval_samples_float))
    if not np.isclose(interval_samples_float, interval_samples, atol=1e-9):
        raise ValueError(
            "interval_s does not correspond to an integer number of samples"
        )
    return first_offset + interval_samples * np.arange(count, dtype=np.int64)


__all__ = [
    "active_channel_to_delay_response",
    "active_to_fft_bins",
    "compensate_cfo",
    "estimate_cfo_from_cp",
    "fft_bins_to_active",
    "find_sequence_start",
    "get_zc_sequence",
    "least_squares_channel_estimate",
    "ofdm_demodulate_symbol",
    "ofdm_modulate_symbol",
    "periodic_sample_offsets",
    "phase_align_channel_estimates",
    "sample_timestamps",
]
