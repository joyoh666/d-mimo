from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from config import SoundingConfig


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ChannelOutput:
    received: ComplexArray
    noiseless_received: ComplexArray
    impulse_responses: ComplexArray
    noise_variance: float


def make_example_channels(cfg: SoundingConfig) -> ComplexArray:
    """Generate six reproducible short multipath impulse responses."""
    rng = np.random.default_rng(cfg.random_seed)
    delays = np.array([0, 2, 5])
    relative_amplitudes = np.array([1.0, 0.30, 0.14])
    channels = np.zeros((cfg.num_tx_branches, delays[-1] + 1), np.complex128)

    for branch in range(cfg.num_tx_branches):
        large_scale = 0.65 + 0.55 * rng.random()
        phases = rng.uniform(-np.pi, np.pi, size=delays.size)
        taps = large_scale * relative_amplitudes * np.exp(1j * phases)
        channels[branch, delays] = taps

    # Keep the first path of the synchronization branch dominant so the
    # correlation peak corresponds to the intended frame boundary.
    b = cfg.sync_tx_branch
    channels[b, 0] = 1.0 + 0.0j
    return channels


def apply_channel(
    branch_samples: ComplexArray,
    impulse_responses: ComplexArray,
    cfg: SoundingConfig,
) -> ChannelOutput:
    """Sum all branches after multipath, then add delay, CFO, and AWGN."""
    if branch_samples.shape[0] != impulse_responses.shape[0]:
        raise ValueError("Branch count and channel count do not match.")

    channel_len = impulse_responses.shape[1]
    convolved_len = branch_samples.shape[1] + channel_len - 1
    summed = np.zeros(convolved_len, dtype=np.complex128)

    for branch in range(branch_samples.shape[0]):
        summed += np.convolve(branch_samples[branch], impulse_responses[branch])

    delayed = np.concatenate(
        (
            np.zeros(cfg.timing_offset_samples, dtype=np.complex128),
            summed,
        )
    )

    n = np.arange(delayed.size)
    cfo_rotation = np.exp(1j * 2.0 * np.pi * cfg.cfo_hz * n / cfg.sample_rate_hz)
    noiseless = delayed * cfo_rotation

    # Define SNR over the active burst rather than over the 5 ms zero padding.
    active_start = cfg.timing_offset_samples
    active_stop = active_start + cfg.burst_samples + channel_len - 1
    active = noiseless[active_start:active_stop]
    signal_power = float(np.mean(np.abs(active) ** 2))
    noise_variance = signal_power / (10.0 ** (cfg.snr_db / 10.0))

    rng = np.random.default_rng(cfg.random_seed + 1)
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(noiseless.size)
        + 1j * rng.standard_normal(noiseless.size)
    )
    return ChannelOutput(
        received=noiseless + noise,
        noiseless_received=noiseless,
        impulse_responses=impulse_responses,
        noise_variance=noise_variance,
    )


def frequency_response_at_bins(
    impulse_responses: ComplexArray,
    centered_bins: tuple[int, ...],
    nfft: int,
) -> ComplexArray:
    """Return H_b[k_b] for each branch and its assigned centered bin."""
    result = np.zeros(len(centered_bins), dtype=np.complex128)
    delay = np.arange(impulse_responses.shape[1])
    for branch, k in enumerate(centered_bins):
        result[branch] = np.sum(
            impulse_responses[branch]
            * np.exp(-1j * 2.0 * np.pi * k * delay / nfft)
        )
    return result

