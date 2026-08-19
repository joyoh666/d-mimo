from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from numpy.typing import NDArray

from config import SoundingConfig


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class TransmitFrame:
    branch_samples: ComplexArray
    zc: ComplexArray
    pilot_symbols: ComplexArray
    frequency_grids: ComplexArray


def zadoff_chu(root: int, length: int) -> ComplexArray:
    """Generate the odd-length Zadoff-Chu definition used by this prototype."""
    if length % 2 == 0:
        raise ValueError("This example implements the odd-length ZC formula.")
    if gcd(root, length) != 1:
        raise ValueError("ZC root and length must be coprime.")

    n = np.arange(length, dtype=np.float64)
    return np.exp(-1j * np.pi * root * n * (n + 1.0) / length)


def centered_bin_to_array_index(bin_index: int, nfft: int) -> int:
    """Map centered bin [-N/2, N/2-1] to an fftshift-ed array index."""
    return bin_index + nfft // 2


def make_pilot_symbols(cfg: SoundingConfig) -> ComplexArray:
    """Create deterministic QPSK pilots with unit combined OFDM power."""
    base = np.array(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j, 1 + 1j, -1 + 1j],
        dtype=np.complex128,
    )[: cfg.num_tx_branches] / np.sqrt(2.0)

    # With unitary IFFT, total time-domain pilot power is sum(|P|^2)/NFFT.
    amplitude = np.sqrt(cfg.nfft / cfg.num_tx_branches)
    return amplitude * base


def build_transmit_frame(cfg: SoundingConfig) -> TransmitFrame:
    """Build one 5 ms frame for every transmit branch.

    Only sync_tx_branch sends the common ZC preamble. Every branch sends its
    pilot on a unique subcarrier. The same OFDM pilot symbol is transmitted
    three times so the receiver can implement the paper's noise averaging.
    """
    cfg.validate()
    zc = zadoff_chu(cfg.zc_root, cfg.zc_len)
    pilots = make_pilot_symbols(cfg)

    branches = np.zeros(
        (cfg.num_tx_branches, cfg.period_samples), dtype=np.complex128
    )
    grids = np.zeros(
        (cfg.num_tx_branches, cfg.nfft), dtype=np.complex128
    )

    branches[cfg.sync_tx_branch, : cfg.zc_len] = zc

    for branch, centered_bin in enumerate(cfg.pilot_bins):
        array_index = centered_bin_to_array_index(centered_bin, cfg.nfft)
        grids[branch, array_index] = pilots[branch]

        # grids are stored in centered/fftshift order.
        useful = (
            np.fft.ifft(np.fft.ifftshift(grids[branch])) * np.sqrt(cfg.nfft)
        )
        with_cp = np.concatenate((useful[-cfg.cp_len :], useful))
        for repetition in range(cfg.pilot_repetitions):
            start = cfg.pilot_symbol_cp_start(repetition)
            branches[branch, start : start + with_cp.size] = with_cp

    return TransmitFrame(
        branch_samples=branches,
        zc=zc,
        pilot_symbols=pilots,
        frequency_grids=grids,
    )

