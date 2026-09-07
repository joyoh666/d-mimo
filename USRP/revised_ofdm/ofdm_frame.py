"""OFDM synchronization sequences, reference pilots, and TX-frame creation."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

try:  # Support script execution and package-style imports.
    from .config import DEFAULT_CONFIG, OFDMConfig
    from ...modulate import modulate
except ImportError:
    from USRP.test.revised_ofdm.config import DEFAULT_CONFIG, OFDMConfig
    from modulate import modulate


ComplexArray = NDArray[np.complex64]
UInt8Array = NDArray[np.uint8]


class FrameArtifacts(TypedDict):
    """Objects produced by build_frame, using the original key names."""

    resource_maps_tx: ComplexArray
    waveform: ComplexArray
    pdsch_idx: tuple[NDArray[np.intp], ...]
    ss_td_with_cp: ComplexArray
    data_bits: UInt8Array


def get_zc_sequence(N: int, q: int = 25) -> ComplexArray:
    """Generate the Zadoff-Chu sequence used in the original PSS mapping."""
    m = np.arange(N)
    phase = -np.pi * q * m * (m + 1) / N
    return (np.cos(phase) + 1j * np.sin(phase)).astype(np.complex64)


def sss_sequence(N: int, q: int) -> ComplexArray:
    """Generate the original deterministic BPSK SSS-like sequence.

    This is not a standards-compliant LTE/NR SSS. ``q`` is used as the random
    seed so that the same known sequence can always be reproduced.
    """
    rng = np.random.default_rng(q)
    sequence = rng.integers(0, 2, size=N)
    return (2 * sequence - 1 + 0.0j).astype(np.complex64)


def generate_known_reference_sequence(
    num_subcarriers: int,
    seed: int = 2026,
) -> ComplexArray:
    """Generate the original deterministic unit-power QPSK reference."""
    rng = np.random.default_rng(seed)
    qpsk_constellation = np.array(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
        dtype=np.complex64,
    ) / np.sqrt(2)
    sequence = rng.choice(qpsk_constellation, size=num_subcarriers)
    sequence = sequence / np.sqrt(np.mean(np.abs(sequence) ** 2))
    return sequence.astype(np.complex64)


def _generate_data_bits(num_bits: int, rng: Generator | None) -> UInt8Array:
    if rng is None:
        # Preserve the original script's global NumPy RNG behavior by default.
        return np.random.randint(0, 2, num_bits, dtype=np.uint8)
    return rng.integers(0, 2, num_bits, dtype=np.uint8)


def build_frame(
    config: OFDMConfig,
    known_ref_seq: ComplexArray,
    *,
    rng: Generator | None = None,
) -> FrameArtifacts:
    """Build the original resource grid and time-domain OFDM waveform.

    The function reads every parameter from ``config``. This makes it possible
    to build different bandwidths without redefining globals in this module.
    Passing ``rng=None`` retains the original global ``np.random`` behavior.
    """
    known_ref_seq = np.asarray(known_ref_seq, dtype=np.complex64)
    if known_ref_seq.shape != (config.N,):
        raise ValueError(
            f"known_ref_seq must have shape ({config.N},), "
            f"got {known_ref_seq.shape}"
        )

    N = config.N
    FFT_SIZE = config.FFT_SIZE
    PDSCH_PLACEHOLDER = np.complex64(999)

    resource_maps = np.full(
        (
            config.num_subframe_per_frame,
            config.num_slot_per_subframe,
            config.num_symbols_per_slot,
            N,
        ),
        PDSCH_PLACEHOLDER,
        dtype=np.complex64,
    )

    # PSS/SSS placement
    zc_seq = get_zc_sequence(config.N_PSS, 25)
    pss_start_idx = N - (config.N_PSS + 1) - (
        (N - (config.N_PSS + 1)) // 2
    )

    pss = np.zeros(N, dtype=np.complex64)
    pss[pss_start_idx:pss_start_idx + config.N_PSS] = zc_seq

    sss_first = np.zeros(N, dtype=np.complex64)
    sss_first[pss_start_idx:pss_start_idx + config.N_PSS] = sss_sequence(
        config.N_PSS, 0
    )
    sss_second = np.zeros(N, dtype=np.complex64)
    sss_second[pss_start_idx:pss_start_idx + config.N_PSS] = sss_sequence(
        config.N_PSS, 1
    )

    for subframe_index in (0, 5):
        resource_maps[subframe_index, 0, 6, :] = pss
    resource_maps[0, 0, 5, :] = sss_first
    resource_maps[5, 0, 5, :] = sss_second

    # Reference symbol 0 of every slot
    known_ref_stacked = np.tile(
        known_ref_seq.reshape(1, 1, N),
        [
            config.num_subframe_per_frame,
            config.num_slot_per_subframe,
            1,
        ],
    )
    resource_maps[..., 0, :] = known_ref_stacked

    # Consecutive identical virtual-pilot symbols for time averaging
    for subframe_index, slot_index, symbol_index in (
        config.get_virtual_pilot_positions()
    ):
        resource_maps[subframe_index, slot_index, symbol_index, :] = known_ref_seq

    # Fill every remaining placeholder RE with PDSCH.
    pdsch_idx = np.where(resource_maps == PDSCH_PLACEHOLDER)
    num_data_symbols = int(pdsch_idx[0].size)
    bits_per_symbol = int(np.log2(config.modulation_order))
    data_bits = _generate_data_bits(num_data_symbols * bits_per_symbol, rng)
    resource_maps[pdsch_idx] = modulate(data_bits, config.modulation_order)

    # Active subcarriers -> FFT bins, with DC left unused
    fft_symbols = np.zeros(
        (
            config.num_subframe_per_frame,
            config.num_slot_per_subframe,
            config.num_symbols_per_slot,
            FFT_SIZE,
        ),
        dtype=np.complex64,
    )
    fft_symbols[..., 1:N // 2 + 1] = resource_maps[..., N // 2:]
    fft_symbols[..., -(N // 2):] = resource_maps[..., :N // 2]

    td_symbols = np.fft.ifft(fft_symbols, axis=-1, norm="ortho").astype(
        np.complex64
    )
    td_symbols_with_cp_0th = np.concatenate(
        [
            td_symbols[:, :, 0:1, -config.first_CP_length:],
            td_symbols[:, :, 0:1, :],
        ],
        axis=-1,
    )
    td_symbols_with_cp_otherwise = np.concatenate(
        [
            td_symbols[:, :, 1:, -config.normal_CP_length:],
            td_symbols[:, :, 1:, :],
        ],
        axis=-1,
    )
    td_symbols_with_cp = np.concatenate(
        [
            np.reshape(
                td_symbols_with_cp_0th,
                (
                    config.num_subframe_per_frame,
                    config.num_slot_per_subframe,
                    -1,
                ),
            ),
            np.reshape(
                td_symbols_with_cp_otherwise,
                (
                    config.num_subframe_per_frame,
                    config.num_slot_per_subframe,
                    -1,
                ),
            ),
        ],
        axis=-1,
    )

    # Synchronization template and digital normalization are unchanged.
    ss_td_with_cp = td_symbols_with_cp_otherwise[0, 0, 4:, :].flatten()
    td_symbols_with_cp *= np.sqrt(config.POWER)
    waveform = np.reshape(td_symbols_with_cp, (1, -1)).astype(np.complex64)

    return {
        "resource_maps_tx": resource_maps,
        "waveform": waveform,
        "pdsch_idx": pdsch_idx,
        "ss_td_with_cp": ss_td_with_cp,
        "data_bits": data_bits,
    }


def build_frame_from_config(
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    rng: Generator | None = None,
) -> tuple[ComplexArray, FrameArtifacts]:
    """Generate the reference sequence and frame from one shared config."""
    known_ref_seq = generate_known_reference_sequence(
        config.N,
        seed=config.effective_reference_sequence_seed(),
    )
    return known_ref_seq, build_frame(config, known_ref_seq, rng=rng)
