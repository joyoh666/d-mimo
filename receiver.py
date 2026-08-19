from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from config import SoundingConfig
from transmitter import centered_bin_to_array_index


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class ReceiveResult:
    sync_start: int
    sync_metric: RealArray
    cfo_estimate_hz: float
    corrected_samples: ComplexArray
    received_grid: ComplexArray
    received_grids: ComplexArray
    per_pilot_channel_estimates: ComplexArray
    channel_estimates: ComplexArray


def normalized_zc_correlation(received: ComplexArray, zc: ComplexArray) -> RealArray:
    """Compute a sliding, power-normalized matched-filter metric."""
    if received.size < zc.size:
        raise ValueError("Received vector is shorter than the ZC sequence.")

    correlation = np.correlate(received, zc, mode="valid")
    window_energy = np.convolve(
        np.abs(received) ** 2,
        np.ones(zc.size, dtype=np.float64),
        mode="valid",
    )
    zc_energy = float(np.sum(np.abs(zc) ** 2))
    return np.abs(correlation) ** 2 / (window_energy * zc_energy + 1e-15)


def estimate_cfo_from_cp(
    received: ComplexArray,
    cp_start: int,
    cfg: SoundingConfig,
) -> float:
    """Estimate CFO from CP correlations across all repeated pilots.

    The unambiguous range is approximately +/- SCS/2.
    """
    # In an isolated burst, early CP samples can contain a transient from the
    # preceding guard. Use the latter half and combine all pilot repetitions.
    cp_skip = cfg.cp_len // 2
    expected_size = cfg.cp_len - cp_skip
    combined_correlation = 0.0j

    for repetition in range(cfg.pilot_repetitions):
        symbol_cp_start = cp_start + repetition * cfg.pilot_symbol_samples
        cp = received[
            symbol_cp_start + cp_skip : symbol_cp_start + cfg.cp_len
        ]
        repeated_tail = received[
            symbol_cp_start + cfg.nfft + cp_skip :
            symbol_cp_start + cfg.nfft + cfg.cp_len
        ]
        if cp.size != expected_size or repeated_tail.size != expected_size:
            raise ValueError("Not enough samples for CP-based CFO estimation.")
        combined_correlation += np.vdot(cp, repeated_tail)

    phase = np.angle(combined_correlation)
    return cfg.sample_rate_hz * phase / (2.0 * np.pi * cfg.nfft)


def estimate_cfo_from_repeated_pilots(
    received: ComplexArray,
    sync_start: int,
    cfg: SoundingConfig,
) -> float:
    """Estimate CFO from phase drift between identical pilot symbols."""
    if cfg.pilot_repetitions < 2:
        cp_start = sync_start + cfg.pilot_cp_start
        return estimate_cfo_from_cp(received, cp_start, cfg)

    combined_correlation = 0.0j
    for repetition in range(cfg.pilot_repetitions - 1):
        first_start = sync_start + cfg.pilot_symbol_data_start(repetition)
        second_start = sync_start + cfg.pilot_symbol_data_start(repetition + 1)
        first = received[first_start : first_start + cfg.nfft]
        second = received[second_start : second_start + cfg.nfft]
        if first.size != cfg.nfft or second.size != cfg.nfft:
            raise ValueError("Not enough samples for repeated-pilot CFO estimation.")
        combined_correlation += np.vdot(first, second)

    phase = np.angle(combined_correlation)
    return (
        cfg.sample_rate_hz
        * phase
        / (2.0 * np.pi * cfg.pilot_symbol_samples)
    )


def receive_frame(
    received: ComplexArray,
    zc: ComplexArray,
    pilot_symbols: ComplexArray,
    cfg: SoundingConfig,
) -> ReceiveResult:
    """Synchronize, correct CFO, demodulate, and estimate sparse CSI.

    One LS estimate is formed from each repeated pilot symbol, then the three
    complex estimates are averaged as described by the prototype manuscript.
    """
    metric = normalized_zc_correlation(received, zc)
    sync_start = int(np.argmax(metric))
    cp_start = sync_start + cfg.pilot_cp_start

    cfo_estimate = estimate_cfo_from_repeated_pilots(
        received,
        sync_start,
        cfg,
    )
    n = np.arange(received.size)
    corrected = received * np.exp(
        -1j * 2.0 * np.pi * cfo_estimate * n / cfg.sample_rate_hz
    )

    grids = []
    for repetition in range(cfg.pilot_repetitions):
        data_start = sync_start + cfg.pilot_symbol_data_start(repetition)
        useful = corrected[data_start : data_start + cfg.nfft]
        if useful.size != cfg.nfft:
            raise ValueError("Not enough samples for repeated OFDM pilots.")
        grid = np.fft.fftshift(np.fft.fft(useful)) / np.sqrt(cfg.nfft)
        grids.append(grid)

    received_grids = np.stack(grids, axis=0)
    received_grid = np.mean(received_grids, axis=0)
    pilot_indices = np.array(
        [centered_bin_to_array_index(k, cfg.nfft) for k in cfg.pilot_bins]
    )
    per_pilot_estimates = (
        received_grids[:, pilot_indices] / pilot_symbols[np.newaxis, :]
    )
    estimates = np.mean(per_pilot_estimates, axis=0)

    return ReceiveResult(
        sync_start=sync_start,
        sync_metric=metric,
        cfo_estimate_hz=float(cfo_estimate),
        corrected_samples=corrected,
        received_grid=received_grid,
        received_grids=received_grids,
        per_pilot_channel_estimates=per_pilot_estimates,
        channel_estimates=estimates,
    )
