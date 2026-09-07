#%%
import sys
import os
import json
from datetime import datetime

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import correlate

from modulate import modulate, modulations
from usrp_utils import sendAndReceive


# =============================================================================
# User-configurable settings
# =============================================================================
carrier_frequency = 5.35e9
Tx_gain, Rx_gain = 30, 30

bandwidth_options = {
    # BW [MHz] : (# active subcarriers, FFT size)
    1.4: (72, 128),
    3:   (180, 256),
    5:   (300, 512),
    10:  (600, 1024),
    15:  (900, 1536),
    20:  (1200, 2048),
    50:  (3000, 4096),
    400: (24000, 32768),
}

# Default mode: run the currently selected bandwidth only.
# To sweep multiple settings, for example:
# CAPTURE_BANDWIDTHS = [5, 10, 20]
BANDWIDTH = 10
CAPTURE_BANDWIDTHS = [BANDWIDTH]

modulation_order = 4
POWER = 4
NUM_CHANNEL_CAPTURES = 50
REFERENCE_SEQUENCE_SEED = 2026
OUTPUT_DIR = "saved_channel_estimates"
NUM_VIRTUAL_PILOTS = 7
VIRTUAL_PILOT_SUBFRAME = 0
VIRTUAL_PILOT_SLOT = 1
VIRTUAL_PILOT_SYMBOL_START = 0
VIRTUAL_PILOT_SYMBOLS = list(range(VIRTUAL_PILOT_SYMBOL_START, VIRTUAL_PILOT_SYMBOL_START + NUM_VIRTUAL_PILOTS))
PHASE_ALIGN_VIRTUAL_PILOTS = True

# LTE-like fixed parameters
DELTA_F = 180e3  # Subcarrier spacing [Hz]
normal_CP_time = 4.7e-6
first_CP_time = 5.2e-6
N_PSS = 62
num_symbols_per_slot = 7
num_slot_per_subframe = 2
num_subframe_per_frame = 10
num_symbols_frame = num_symbols_per_slot * num_slot_per_subframe * num_subframe_per_frame


# =============================================================================
# Helper functions
# =============================================================================
def get_zc_sequence(N, q=25):
    m = np.arange(N)
    seq = -np.pi * q * m * (m + 1) / N
    i = np.cos(seq)
    q_part = np.sin(seq)
    return (i + 1j * q_part).astype(np.complex64)


def sss_sequence(N, q):
    rng = np.random.default_rng(q)
    sequence = rng.integers(0, 2, size=N)
    return (2 * sequence - 1 + 0.0j).astype(np.complex64)


def generate_known_reference_sequence(num_subcarriers, seed=2026):
    """
    Generate a known random reference sequence with unit average power,
    similar to constant-power reference/pilot symbols.
    Here we use a deterministic seeded QPSK sequence.
    """
    rng = np.random.default_rng(seed)
    qpsk_constellation = np.array(
        [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
        dtype=np.complex64,
    ) / np.sqrt(2)
    seq = rng.choice(qpsk_constellation, size=num_subcarriers)

    # Normalize just in case to exactly unit average power.
    seq = seq / np.sqrt(np.mean(np.abs(seq) ** 2))
    return seq.astype(np.complex64)


def get_virtual_pilot_positions():
    return [
        (VIRTUAL_PILOT_SUBFRAME, VIRTUAL_PILOT_SLOT, sym)
        for sym in VIRTUAL_PILOT_SYMBOLS
    ]


def phase_align_channel_estimates(h_virtual_pilots):
    """Align common phase between repeated pilot-based estimates before averaging."""
    ref = h_virtual_pilots[0]
    aligned = [ref.astype(np.complex64)]
    common_phases = [0.0]
    for idx in range(1, h_virtual_pilots.shape[0]):
        h_i = h_virtual_pilots[idx]
        phase = np.angle(np.vdot(ref, h_i))
        aligned_h = h_i * np.exp(-1j * phase)
        aligned.append(aligned_h.astype(np.complex64))
        common_phases.append(float(phase))
    return np.stack(aligned, axis=0).astype(np.complex64), np.asarray(common_phases, dtype=np.float32)


def _pilot_snr_db_from_equalized(equalized_pilots_fd, known_ref_seq):
    if equalized_pilots_fd.size == 0:
        return 0.0
    equalized_unit = equalized_pilots_fd / known_ref_seq.reshape(1, -1).astype(np.complex64)
    mse = np.mean(np.abs(equalized_unit - 1.0) ** 2)
    return float(10.0 * np.log10(1.0 / (mse + 1e-15)))


def get_system_params(bandwidth_mhz):
    N, FFT_SIZE = bandwidth_options[bandwidth_mhz]
    sampling_rate = DELTA_F * FFT_SIZE
    T = 1 / DELTA_F

    normal_CP_length = round(normal_CP_time * sampling_rate)
    first_CP_length = round(first_CP_time * sampling_rate)

    slot_length = (
        FFT_SIZE * num_symbols_per_slot
        + normal_CP_length * (num_symbols_per_slot - 1)
        + first_CP_length
    )
    frame_length = slot_length * num_slot_per_subframe * num_subframe_per_frame

    return {
        "bandwidth_mhz": bandwidth_mhz,
        "N": N,
        "FFT_SIZE": FFT_SIZE,
        "delta_f": DELTA_F,
        "sampling_rate": sampling_rate,
        "T": T,
        "normal_CP_length": normal_CP_length,
        "first_CP_length": first_CP_length,
        "slot_length": slot_length,
        "frame_length": frame_length,
        "carrier_frequency": carrier_frequency,
        "Tx_gain": Tx_gain,
        "Rx_gain": Rx_gain,
        "modulation_order": modulation_order,
        "POWER": POWER,
        "num_symbols_per_slot": num_symbols_per_slot,
        "num_slot_per_subframe": num_slot_per_subframe,
        "num_subframe_per_frame": num_subframe_per_frame,
        "num_symbols_frame": num_symbols_frame,
        "N_PSS": N_PSS,
        "num_virtual_pilots": NUM_VIRTUAL_PILOTS,
        "virtual_pilot_positions": get_virtual_pilot_positions(),
    }


def build_frame(params, known_ref_seq):
    N = params["N"]
    FFT_SIZE = params["FFT_SIZE"]
    first_CP_length = params["first_CP_length"]
    normal_CP_length = params["normal_CP_length"]

    PDSCH_PLACEHOLDER = 999

    resource_maps = np.ones(
        (num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, N),
        dtype=np.complex64,
    ) * PDSCH_PLACEHOLDER

    # PSS / SSS placement
    zc_seq = get_zc_sequence(N_PSS, 25)
    pss_start_idx = N - (N_PSS + 1) - ((N - (N_PSS + 1)) // 2)

    pss = np.zeros(N, np.complex64)
    pss[pss_start_idx:pss_start_idx + N_PSS] = zc_seq

    sss_first = np.zeros(N, np.complex64)
    sss_first[pss_start_idx:pss_start_idx + N_PSS] = sss_sequence(N_PSS, 0)

    sss_second = np.zeros(N, np.complex64)
    sss_second[pss_start_idx:pss_start_idx + N_PSS] = sss_sequence(N_PSS, 1)

    for sfn in [0, 5]:
        resource_maps[sfn, 0, 6, :] = pss

    resource_maps[0, 0, 5, :] = sss_first
    resource_maps[5, 0, 5, :] = sss_second

    # Known random reference sequence mapped on symbol 0 of each slot.
    known_ref_stacked = np.tile(
        known_ref_seq.reshape(1, 1, N),
        [num_subframe_per_frame, num_slot_per_subframe, 1],
    )
    resource_maps[..., 0, :] = known_ref_stacked

    # Copy the same known reference symbol across consecutive symbols to create one
    # virtual pilot estimate through averaging.
    extra_virtual_pilot_symbols = 0
    for sf_idx, slot_idx, sym_idx in get_virtual_pilot_positions():
        if not (0 <= sf_idx < num_subframe_per_frame and 0 <= slot_idx < num_slot_per_subframe):
            continue
        if not (0 <= sym_idx < num_symbols_per_slot):
            continue
        resource_maps[sf_idx, slot_idx, sym_idx, :] = known_ref_seq
        if sym_idx != 0:
            extra_virtual_pilot_symbols += 1

    # PDSCH generation for the remaining REs.
    num_data_symbols = (
        (num_symbols_per_slot - 1)
        * num_slot_per_subframe
        * num_subframe_per_frame
        - 4
        - extra_virtual_pilot_symbols
    ) * N

    data_bits = np.random.randint(
        0,
        2,
        num_data_symbols * int(np.log2(modulation_order)),
        dtype=np.uint8,
    )
    iq = modulate(data_bits, modulation_order)

    pdsch_idx = np.where(resource_maps == PDSCH_PLACEHOLDER)
    resource_maps[pdsch_idx] = iq

    # Map active subcarriers into FFT bins (DC unused)
    fft_symbols = np.zeros(
        (num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, FFT_SIZE),
        np.complex64,
    )
    fft_symbols[..., 1:N // 2 + 1] = resource_maps[..., N // 2:]
    fft_symbols[..., -(N // 2):] = resource_maps[..., :N // 2]

    # OFDM modulation + CP insertion
    td_symbols = np.fft.ifft(fft_symbols, axis=-1, norm="ortho").astype(np.complex64)

    td_symbols_with_cp_0th = np.concatenate(
        [
            td_symbols[:, :, 0:1, -first_CP_length:],
            td_symbols[:, :, 0:1, :],
        ],
        axis=-1,
    )
    td_symbols_with_cp_otherwise = np.concatenate(
        [
            td_symbols[:, :, 1:, -normal_CP_length:],
            td_symbols[:, :, 1:, :],
        ],
        axis=-1,
    )
    td_symbols_with_cp = np.concatenate(
        [
            np.reshape(
                td_symbols_with_cp_0th,
                (num_subframe_per_frame, num_slot_per_subframe, -1),
            ),
            np.reshape(
                td_symbols_with_cp_otherwise,
                (num_subframe_per_frame, num_slot_per_subframe, -1),
            ),
        ],
        axis=-1,
    )

    # Sync sequence for timing estimation (same as original flow)
    ss_td_with_cp = td_symbols_with_cp_otherwise[0, 0, 4:, :].flatten()

    # Digital normalization
    td_symbols_with_cp *= np.sqrt(POWER)
    waveform = np.reshape(td_symbols_with_cp, (1, -1)).astype(np.complex64)

    return {
        "resource_maps_tx": resource_maps,
        "waveform": waveform,
        "pdsch_idx": pdsch_idx,
        "ss_td_with_cp": ss_td_with_cp,
        "data_bits": data_bits,
    }


def receive_and_process(
    usrp,
    params,
    waveform,
    ss_td_with_cp,
    pdsch_idx,
    known_ref_seq,
):
    N = params["N"]
    FFT_SIZE = params["FFT_SIZE"]
    sampling_rate = params["sampling_rate"]
    frame_length = params["frame_length"]
    slot_length = params["slot_length"]
    first_CP_length = params["first_CP_length"]
    normal_CP_length = params["normal_CP_length"]

    frame_rcv = sendAndReceive(
        usrp,
        waveform,
        1,
        carrier_frequency,
        sampling_rate,
        Tx_gain,
        Rx_gain,
        [0],
        [0],
        wait_time=0.2,
        tx_delay_samples=int(sampling_rate * 1e-4),
        rx_trailing_samples=int(sampling_rate * 1e-3),
        otw_format="sc16",
    )
    frame_rcv = frame_rcv[:, 10:]
    frame_rcv -= np.mean(frame_rcv)
    frame_rcv = frame_rcv[0].flatten()

    # Integer FO sync left disabled as in original code
    frame_rcv_ifosync = frame_rcv

    # Timing synchronization
    corr = np.abs(correlate(frame_rcv_ifosync, ss_td_with_cp, "valid", "fft"))
    ss_start_idx = FFT_SIZE * 5 + first_CP_length + normal_CP_length * 4
    sync_idx = int(np.argmax(corr) - ss_start_idx)

    if sync_idx + frame_length > frame_rcv.size or sync_idx < 0:
        print("sync not found, using sync_idx = 0")
        sync_idx = 0

    frame_rcv_timesync = frame_rcv_ifosync[sync_idx:sync_idx + frame_length]
    if frame_rcv_timesync.size < frame_length:
        pad = frame_length - frame_rcv_timesync.size
        frame_rcv_timesync = np.concatenate(
            [frame_rcv_timesync.astype(np.complex64), np.zeros(pad, dtype=np.complex64)],
            axis=0,
        )

    # Fractional frequency offset synchronization by CP correlation
    td_symbols_rcv = np.reshape(
        frame_rcv_timesync,
        (num_subframe_per_frame * num_slot_per_subframe, -1),
    )

    first_CPs = td_symbols_rcv[:, :first_CP_length]
    first_signal_for_CPs = td_symbols_rcv[:, FFT_SIZE:FFT_SIZE + first_CP_length]

    td_symbols_rcv_wo_first = td_symbols_rcv[:, FFT_SIZE + first_CP_length:]
    td_symbols_rcv_wo_first = np.reshape(
        td_symbols_rcv_wo_first,
        (num_subframe_per_frame * num_slot_per_subframe * (num_symbols_per_slot - 1), -1),
    )
    other_CPs = td_symbols_rcv_wo_first[:, :normal_CP_length]
    other_signal_for_CPs = td_symbols_rcv_wo_first[:, -normal_CP_length:]

    phase_diff = np.angle(
        np.sum(np.conj(other_CPs) * other_signal_for_CPs)
        + np.sum(np.conj(first_CPs) * first_signal_for_CPs)
    )
    sample_idx_diff = FFT_SIZE
    ffo = phase_diff * sampling_rate / (2 * np.pi * sample_idx_diff)

    n = np.arange(frame_rcv_timesync.size)
    compensation_ffo = np.exp(-1j * ffo * n / sampling_rate * 2 * np.pi).astype(np.complex64)
    frame_rcv_ffosync = frame_rcv_timesync * compensation_ffo

    # Residual frequency offset estimation with known reference symbol (symbol 0)
    refsym_rcv_td = np.reshape(
        frame_rcv_ffosync,
        (num_subframe_per_frame * num_slot_per_subframe, -1),
    )
    refsym_rcv_td = refsym_rcv_td[:, first_CP_length:first_CP_length + FFT_SIZE]

    phase_diff = np.angle(
        np.sum(np.conj(refsym_rcv_td[:-1]) * refsym_rcv_td[1:], axis=-1)
    )
    sample_idx_diff = slot_length
    rfo = phase_diff * sampling_rate / (2 * np.pi * sample_idx_diff)
    rfo = np.concatenate([rfo, rfo[-1:]], axis=0)
    rfo = np.expand_dims(rfo, axis=1)

    n = np.expand_dims(np.arange(slot_length), axis=0)
    compensation_rfo = np.exp(-1j * rfo * n / sampling_rate * 2 * np.pi).astype(np.complex64)
    compensation_rfo = compensation_rfo.flatten()
    frame_rcv_rfosync = frame_rcv_ffosync * compensation_rfo

    # Frequency-domain demodulation
    fft_symbols_rcv = np.reshape(
        frame_rcv_rfosync,
        (num_subframe_per_frame * num_slot_per_subframe, -1),
    )
    fft_symbols_rcv = fft_symbols_rcv[:, first_CP_length - normal_CP_length:]
    fft_symbols_rcv = np.reshape(
        fft_symbols_rcv,
        (num_subframe_per_frame * num_slot_per_subframe * num_symbols_per_slot, -1),
    )
    fft_symbols_rcv = fft_symbols_rcv[:, normal_CP_length:]

    fft_symbols_rcv = np.fft.fft(fft_symbols_rcv, axis=-1, norm="ortho")
    fft_symbols_rcv = np.reshape(
        fft_symbols_rcv,
        (num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, FFT_SIZE),
    )

    resource_maps_rcv = np.zeros(
        (num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, N),
        dtype=np.complex64,
    )
    resource_maps_rcv[..., N // 2:] = fft_symbols_rcv[..., 1:N // 2 + 1]
    resource_maps_rcv[..., :N // 2] = fft_symbols_rcv[..., -(N // 2):]

    # Channel estimation using consecutive identical pilots.
    h_virtual_pilots = []
    virtual_pilot_symbols = []
    for sf_idx, slot_idx, sym_idx in get_virtual_pilot_positions():
        if not (0 <= sf_idx < num_subframe_per_frame and 0 <= slot_idx < num_slot_per_subframe):
            continue
        if not (0 <= sym_idx < num_symbols_per_slot):
            continue
        h_sym = resource_maps_rcv[sf_idx, slot_idx, sym_idx, :] / known_ref_seq
        h_virtual_pilots.append(h_sym)
        virtual_pilot_symbols.append(resource_maps_rcv[sf_idx, slot_idx, sym_idx, :])
        # Keep track of where these virtual pilots were extracted from.
        virtual_pilot_positions_used = (sf_idx, slot_idx, sym_idx)

    if not h_virtual_pilots:
        h_virtual_pilots = [resource_maps_rcv[..., 0, :] / known_ref_seq]
        virtual_pilot_symbols = [resource_maps_rcv[..., 0, :]]
        virtual_pilot_positions_used = None

    h_virtual_pilots = np.stack(h_virtual_pilots, axis=0)
    h_fd_virtual_avg_raw = np.mean(h_virtual_pilots, axis=0).astype(np.complex64)
    h_virtual_pilots_aligned, virtual_pilot_common_phases = phase_align_channel_estimates(h_virtual_pilots)
    if PHASE_ALIGN_VIRTUAL_PILOTS:
        h_fd_virtual_avg = np.mean(h_virtual_pilots_aligned, axis=0).astype(np.complex64)
    else:
        h_fd_virtual_avg = h_fd_virtual_avg_raw.copy()

    # Per-subcarrier estimator variance across virtual pilots.
    virtual_pilot_variance_raw = np.mean(
        np.abs(h_virtual_pilots - h_fd_virtual_avg_raw[None, :]) ** 2,
        axis=0,
    ).astype(np.float32)
    virtual_pilot_variance_aligned = np.mean(
        np.abs(h_virtual_pilots_aligned - h_fd_virtual_avg[None, :]) ** 2,
        axis=0,
    ).astype(np.float32)

    # Keep original behavior (one channel estimate for data) but from virtual-averaged pilot.
    h_fd = np.broadcast_to(
        h_fd_virtual_avg.reshape(1, 1, -1),
        (num_subframe_per_frame, num_slot_per_subframe, N),
    ).copy()

    # Baseline single-pilot estimate from the first virtual pilot symbol.
    h_single_fd = h_virtual_pilots[0]
    h_single_fd_full = np.broadcast_to(
        h_single_fd.reshape(1, 1, -1),
        (num_subframe_per_frame, num_slot_per_subframe, N),
    ).copy()

    # Equalize with virtual-averaged and single-pilot estimates.
    h_expand = h_fd[:, :, np.newaxis, :]
    resource_maps_rcv_zf = resource_maps_rcv / (h_expand + 1e-12)
    iq_rcv = resource_maps_rcv_zf[pdsch_idx]

    h_expand_single = h_single_fd_full[:, :, np.newaxis, :]
    resource_maps_rcv_zf_single = resource_maps_rcv / (h_expand_single + 1e-12)
    iq_rcv_single = resource_maps_rcv_zf_single[pdsch_idx]

    # Correct common constellation rotation from the pilot-domain equalizer residual.
    if virtual_pilot_positions_used is None:
        correction_phase = 0.0
    else:
        sfn_ref, slot_ref, _ = virtual_pilot_positions_used
        pilot_eq_ref = resource_maps_rcv_zf[sfn_ref, slot_ref, 0, :] / known_ref_seq
        correction_phase = float(np.angle(np.mean(pilot_eq_ref)))
    correction = np.exp(-1j * correction_phase).astype(np.complex64)
    iq_rcv *= correction
    iq_rcv_single *= correction

    virtual_pilot_received = np.stack(virtual_pilot_symbols, axis=0)
    h_gain_raw = virtual_pilot_received / h_fd_virtual_avg.reshape(1, -1)
    h_gain_single = virtual_pilot_received / h_single_fd.reshape(1, -1)
    single_pilot_snr_db = _pilot_snr_db_from_equalized(h_gain_single, known_ref_seq)
    virtual_pilot_snr_db = _pilot_snr_db_from_equalized(h_gain_raw, known_ref_seq)
    snr_gain_db = virtual_pilot_snr_db - single_pilot_snr_db

    return {
        "frame_rcv": frame_rcv,
        "frame_rcv_timesync": frame_rcv_timesync,
        "frame_rcv_rfosync": frame_rcv_rfosync,
        "resource_maps_rcv": resource_maps_rcv,
        "h_fd": h_fd.astype(np.complex64),
        "h_single_fd": h_single_fd_full.astype(np.complex64),
        "h_fd_virtual_avg_raw": h_fd_virtual_avg_raw.astype(np.complex64),
        "h_fd_virtual_avg_aligned": h_fd_virtual_avg.astype(np.complex64),
        "h_virtual_pilots_fd": h_virtual_pilots,
        "h_virtual_pilots_fd_aligned": h_virtual_pilots_aligned,
        "virtual_pilot_variance_raw": virtual_pilot_variance_raw,
        "virtual_pilot_variance_aligned": virtual_pilot_variance_aligned,
        "virtual_pilot_common_phases_rad": virtual_pilot_common_phases,
        "single_pilot_snr_db": single_pilot_snr_db,
        "virtual_pilot_snr_db": virtual_pilot_snr_db,
        "virtual_pilot_snr_gain_db": snr_gain_db,
        "virtual_pilot_correction_phase_rad": float(correction_phase),
        "iq_rcv": iq_rcv.astype(np.complex64),
        "iq_rcv_single": iq_rcv_single.astype(np.complex64),
        "sync_idx": sync_idx,
        "ffo_hz": float(ffo),
        "rfo_hz_per_slot": rfo.flatten().astype(np.float32),
        "timing_correlation_peak": float(np.max(corr)) if corr.size > 0 else 0.0,
    }


def active_to_full_spectrum(channel_active, N, FFT_SIZE):
    full_spectrum = np.zeros(FFT_SIZE, dtype=np.complex64)
    full_spectrum[1:N // 2 + 1] = channel_active[N // 2:]
    full_spectrum[-(N // 2):] = channel_active[:N // 2]
    return full_spectrum


def channel_to_delay_response(channel_active, N, FFT_SIZE):
    full_spectrum = active_to_full_spectrum(channel_active, N, FFT_SIZE)
    cir = np.fft.ifft(full_spectrum, n=FFT_SIZE, norm="ortho")
    return cir.astype(np.complex64), full_spectrum.astype(np.complex64)


def _delay_residual_power_db(cir, guard_taps=2):
    """Return (main_tap_power, residual_power, residual_to_main_db)."""
    power = np.abs(cir) ** 2
    if power.size == 0:
        return 0.0, 0.0, -np.inf
    peak_idx = int(np.argmax(power))
    mask = np.ones(power.shape[0], dtype=np.bool_)
    lo = max(0, peak_idx - guard_taps)
    hi = min(power.shape[0], peak_idx + guard_taps + 1)
    mask[lo:hi] = False

    main_power = float(np.mean(power[~mask]) + 1e-15)
    residual_power = float(np.mean(power[mask]) + 1e-15)
    residual_reduction_db = 10.0 * np.log10(main_power / residual_power)
    return main_power, residual_power, residual_reduction_db


def save_results(output_dir, params, known_ref_seq, results):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bw_tag = f"BW_{str(params['bandwidth_mhz']).replace('.', 'p')}MHz"

    npz_path = os.path.join(output_dir, f"channel_dataset_{bw_tag}_{timestamp}.npz")
    json_path = os.path.join(output_dir, f"channel_dataset_{bw_tag}_{timestamp}_metadata.json")
    fig_path = os.path.join(output_dir, f"channel_plots_{bw_tag}_{timestamp}.png")

    metadata = {
        "timestamp": timestamp,
        "bandwidth_mhz": params["bandwidth_mhz"],
        "subcarrier_spacing_hz": params["delta_f"],
        "num_active_subcarriers": params["N"],
        "fft_size": params["FFT_SIZE"],
        "sampling_rate_hz": params["sampling_rate"],
        "carrier_frequency_hz": params["carrier_frequency"],
        "tx_gain_db": params["Tx_gain"],
        "rx_gain_db": params["Rx_gain"],
        "num_symbols_per_slot": params["num_symbols_per_slot"],
        "num_slots_per_subframe": params["num_slot_per_subframe"],
        "num_subframes_per_frame": params["num_subframe_per_frame"],
        "num_symbols_per_frame": params["num_symbols_frame"],
        "normal_cp_length_samples": params["normal_CP_length"],
        "first_cp_length_samples": params["first_CP_length"],
        "slot_length_samples": params["slot_length"],
        "frame_length_samples": params["frame_length"],
        "num_channel_captures": int(results["channel_estimates_fd"].shape[0]),
        "saved_channel_tensor_shape": list(results["channel_estimates_fd"].shape),
        "saved_mean_channel_shape": list(results["channel_mean_fd"].shape),
        "saved_delay_response_shape": list(results["channel_impulse_response_td"].shape),
        "saved_single_pilot_delay_response_shape": list(results["channel_impulse_response_td_single_pilot"].shape),
        "samples_per_channel_vector": params["N"],
        "samples_per_delay_response": params["FFT_SIZE"],
        "reference_sequence_type": "seeded_unit_power_qpsk",
        "reference_sequence_seed": REFERENCE_SEQUENCE_SEED,
        "virtual_pilot_count": params["num_virtual_pilots"],
        "virtual_pilot_positions": params["virtual_pilot_positions"],
        "phase_align_virtual_pilots": bool(PHASE_ALIGN_VIRTUAL_PILOTS),
        "expected_channel_estimation_snr_gain_db": float(10 * np.log10(params["num_virtual_pilots"])),
        "achieved_channel_estimation_snr_gain_db_mean": float(np.mean(results["virtual_pilot_snr_gain_db"])),
        "mean_single_to_averaged_residual_delay_gain_db": float(np.mean(results["delay_profile_residual_gain_db"])),
        "modulation_order": params["modulation_order"],
        "digital_power_scale": params["POWER"],
    }

    np.savez_compressed(
        npz_path,
        channel_estimates_fd=results["channel_estimates_fd"],
        channel_estimates_fd_single_pilot=results["channel_estimates_fd_single_pilot"],
        channel_estimates_fd_raw_virtual=results["channel_estimates_fd_raw_virtual"],
        channel_estimates_fd_aligned_virtual=results["channel_estimates_fd_aligned_virtual"],
        channel_mean_fd=results["channel_mean_fd"],
        virtual_pilot_variance_raw=results["virtual_pilot_variance_raw"],
        virtual_pilot_variance_aligned=results["virtual_pilot_variance_aligned"],
        virtual_pilot_common_phases_rad=results["virtual_pilot_common_phases_rad"],
        channel_impulse_response_td=results["channel_impulse_response_td"],
        channel_impulse_response_td_single_pilot=results["channel_impulse_response_td_single_pilot"],
        channel_full_spectrum_fd_single_pilot=results["channel_full_spectrum_fd_single_pilot"],
        channel_full_spectrum_fd=results["channel_full_spectrum_fd"],
        known_reference_sequence=known_ref_seq.astype(np.complex64),
        sync_indices=results["sync_indices"],
        ffo_estimates_hz=results["ffo_estimates_hz"],
        rfo_estimates_hz=results["rfo_estimates_hz"],
        timing_correlation_peaks=results["timing_correlation_peaks"],
        single_pilot_snr_db=results["single_pilot_snr_db"],
        virtual_pilot_snr_db=results["virtual_pilot_snr_db"],
        virtual_pilot_snr_gain_db=results["virtual_pilot_snr_gain_db"],
        delay_profile_residual_gain_db=results["delay_profile_residual_gain_db"],
        virtual_pilot_correction_phase_rad=results["virtual_pilot_correction_phase_rad"],
        example_rx_constellation=results["example_iq_rcv"],
        example_rx_frame=results["example_frame_rcv"],
        metadata_json=json.dumps(metadata),
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return npz_path, json_path, fig_path, metadata


def plot_results(fig_path, params, results):
    N = params["N"]
    FFT_SIZE = params["FFT_SIZE"]
    sampling_rate = params["sampling_rate"]

    channel_mean_fd = results["channel_mean_fd"]
    channel_impulse_response_td = results["channel_impulse_response_td"]
    channel_impulse_response_td_single_pilot = results["channel_impulse_response_td_single_pilot"]
    example_iq_rcv = results["example_iq_rcv"]
    example_iq_rcv_single = results.get("example_iq_rcv_single", example_iq_rcv)
    example_frame_rcv = results["example_frame_rcv"]
    single_pilot_snr_db = results["single_pilot_snr_db"]
    virtual_pilot_snr_db = results["virtual_pilot_snr_db"]
    virtual_pilot_snr_gain_db = results["virtual_pilot_snr_gain_db"]
    virtual_pilot_variance_raw = results["virtual_pilot_variance_raw"]
    virtual_pilot_variance_aligned = results["virtual_pilot_variance_aligned"]
    delay_profile_residual_gain_db = results.get("delay_profile_residual_gain_db", None)

    avg_abs_H = np.mean(np.abs(channel_mean_fd), axis=0)
    avg_abs_h_delay = np.mean(np.abs(channel_impulse_response_td), axis=0)
    avg_abs_h_delay_single = np.mean(
        np.abs(channel_impulse_response_td_single_pilot), axis=0
    )
    avg_var_raw = np.mean(virtual_pilot_variance_raw, axis=0)
    avg_var_aligned = np.mean(virtual_pilot_variance_aligned, axis=0)
    avg_gain_db = 10 * np.log10((avg_var_raw + 1e-15) / (avg_var_aligned / max(params["num_virtual_pilots"], 1) + 1e-15))
    measured_gain_db = float(np.mean(virtual_pilot_snr_gain_db))
    single_main_power, single_residual_power, _ = _delay_residual_power_db(
        avg_abs_h_delay_single,
        guard_taps=max(1, int(np.ceil(params["FFT_SIZE"] / 500))),
    )
    avg_main_power, avg_residual_power, _ = _delay_residual_power_db(
        avg_abs_h_delay,
        guard_taps=max(1, int(np.ceil(params["FFT_SIZE"] / 500))),
    )
    measured_delay_gain_db = 10.0 * np.log10((single_residual_power + 1e-15) / (avg_residual_power + 1e-15))
    if delay_profile_residual_gain_db is not None:
        measured_delay_gain_db = float(np.mean(delay_profile_residual_gain_db))

    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(
        f"Channel capture summary | BW={params['bandwidth_mhz']} MHz | "
        f"N={N} | FFT={FFT_SIZE} | captures={channel_mean_fd.shape[0]} | "
        f"VP={params['num_virtual_pilots']} | measured pilot gain={measured_gain_db:.2f} dB"
    )

    axs[0, 0].plot(avg_abs_H)
    axs[0, 0].set_title("Average |H[k]| across captures")
    axs[0, 0].set_xlabel("Subcarrier index")
    axs[0, 0].set_ylabel("Magnitude")
    axs[0, 0].grid(True, alpha=0.3)

    im = axs[0, 1].imshow(
        np.abs(channel_mean_fd),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    axs[0, 1].set_title("|H[k]| for repeated captures")
    axs[0, 1].set_xlabel("Subcarrier index")
    axs[0, 1].set_ylabel("Capture index")
    fig.colorbar(im, ax=axs[0, 1], fraction=0.046, pad=0.04)

    axs[0, 2].plot(avg_abs_h_delay)
    axs[0, 2].set_title("Average |h[n]| in delay domain (IFFT of channel)")
    axs[0, 2].set_xlabel("Delay sample")
    axs[0, 2].set_ylabel("Magnitude")
    axs[0, 2].grid(True, alpha=0.3)

    fig_delay, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(avg_abs_h_delay_single, label="Single-pilot delay |h[n]|")
    axes[0].plot(avg_abs_h_delay, linestyle="--", label="7-pilot averaged delay |h[n]|")
    axes[0].set_title("Delay-domain mean profile")
    axes[0].set_xlabel("Delay sample")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].text(
        0.05,
        0.82,
        f"Single residual power: {10*np.log10(single_residual_power + 1e-15):.2f} dB",
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.05,
        0.65,
        f"Averaged residual power: {10*np.log10(avg_residual_power + 1e-15):.2f} dB",
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.05,
        0.48,
        f"Residual suppression gain: {(10*np.log10(single_residual_power + 1e-15) - 10*np.log10(avg_residual_power + 1e-15)):.2f} dB",
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.05,
        0.31,
        f"Delay mean |h| peak: single={single_main_power:.3e}, averaged={avg_main_power:.3e}",
        transform=axes[1].transAxes,
    )
    axes[1].axis("off")
    axes[1].set_title("Delay-domain residual metric")
    delay_fig_path = fig_path.replace("channel_plots_", "channel_plots_delay_")
    fig_delay.suptitle(f"Delay profile comparison | measured delay residual suppression={measured_delay_gain_db:.2f} dB")
    plt.tight_layout()
    plt.savefig(delay_fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig_delay)

    axs[1, 0].plot(10 * np.log10(avg_var_raw + 1e-15), label="Raw virtual-pilot variance")
    axs[1, 0].plot(10 * np.log10(avg_var_aligned + 1e-15), label="Phase-aligned virtual-pilot variance")
    axs[1, 0].set_title("Virtual pilot estimator variance")
    axs[1, 0].set_xlabel("Subcarrier index")
    axs[1, 0].set_ylabel("Variance (dB)")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    axs[1, 1].plot(10 * np.log10(avg_var_raw + 1e-15), label="Raw pilot variance")
    axs[1, 1].plot(10 * np.log10(avg_var_aligned + 1e-15), label="Aligned pilot variance")
    axs[1, 1].plot(10 * np.log10(avg_gain_db + 1e-15), label="Estimated gain from averaging")
    axs[1, 1].set_title("Variance reduction from averaging")
    axs[1, 1].set_xlabel("Subcarrier index")
    axs[1, 1].set_ylabel("dB")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    axs[1, 2].psd(
        example_frame_rcv,
        NFFT=example_frame_rcv.size,
        Fs=sampling_rate,
        scale_by_freq=False,
    )
    axs[1, 2].set_title("Example received-frame spectrum")
    axs[1, 2].set_xlabel("Frequency (Hz)")
    axs[1, 2].set_ylabel("Power spectrum (dB)")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    fig2, axes = plt.subplots(1, 2, figsize=(12, 6))
    real_gt = np.real(modulations[modulation_order])
    peak_amp = 2 * np.max(np.abs(real_gt))
    axes[0].scatter(np.real(example_iq_rcv_single), np.imag(example_iq_rcv_single), s=1.0, marker="o")
    axes[0].set_xlim([-peak_amp, peak_amp])
    axes[0].set_ylim([-peak_amp, peak_amp])
    axes[0].set_xlabel("In-Phase")
    axes[0].set_ylabel("Quadrature")
    axes[0].set_title("Single-pilot equalized constellation")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(np.real(example_iq_rcv), np.imag(example_iq_rcv), s=1.0, marker="o")
    axes[1].set_xlim([-peak_amp, peak_amp])
    axes[1].set_ylim([-peak_amp, peak_amp])
    axes[1].set_xlabel("In-Phase")
    axes[1].set_ylabel("Quadrature")
    axes[1].set_title("7-pilot averaged equalized constellation")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig2)

    eq_psd_path = fig_path.replace("channel_plots_", "channel_plots_eqpsd_")
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].psd(
        example_iq_rcv_single,
        NFFT=min(
            4096,
            max(
                256,
                1 << int(np.ceil(np.log2(max(1, example_iq_rcv_single.size)))),
            ),
        ),
        Fs=sampling_rate,
        scale_by_freq=False,
    )
    axes[0].set_title("Single-pilot equalized PSD")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power spectrum (dB)")
    axes[0].grid(True, alpha=0.3)

    axes[1].psd(
        example_iq_rcv,
        NFFT=min(
            4096,
            max(
                256,
                1 << int(np.ceil(np.log2(max(1, example_iq_rcv.size)))),
            ),
        ),
        Fs=sampling_rate,
        scale_by_freq=False,
    )
    axes[1].set_title("7-pilot averaged equalized PSD")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power spectrum (dB)")
    axes[1].grid(True, alpha=0.3)

    fig3.suptitle(
        f"Equalized spectrum | measured equalization gain={measured_gain_db:.2f} dB | "
        f"VP={params['num_virtual_pilots']}"
    )
    plt.tight_layout()
    plt.savefig(eq_psd_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig3)

    return {
        "channel_plots": fig_path,
        "delay_profile": delay_fig_path,
        "equalized_psd": eq_psd_path,
    }


# =============================================================================
# Main execution
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # USRP initialization
    usrp = uhd.usrp.MultiUSRP(
        "addr0=192.168.10.2,second_addr=192.168.11.2,third_addr=192.168.12.2,fourth_addr=192.168.13.2"
    )
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    print("USRP loaded. Session Ready.")

    usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0"), 0)
    usrp.set_tx_antenna("TX/RX", 0)

    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("B:0"), 0)
    usrp.set_rx_antenna("TX/RX", 0)

    print("Tx:")
    print(usrp.get_tx_subdev_spec(0))
    print("Rx:")
    print(usrp.get_rx_subdev_spec(0))

    for bandwidth_mhz in CAPTURE_BANDWIDTHS:
        print("=" * 80)
        print(f"Running bandwidth setting: {bandwidth_mhz} MHz")
        print(
            f"Virtual pilot count: {NUM_VIRTUAL_PILOTS} | "
            f"expected channel-estimation gain: {10 * np.log10(NUM_VIRTUAL_PILOTS):.2f} dB"
        )

        params = get_system_params(bandwidth_mhz)
        known_ref_seq = generate_known_reference_sequence(
            params["N"],
            seed=REFERENCE_SEQUENCE_SEED + int(10 * bandwidth_mhz),
        )

        tx_objects = build_frame(params, known_ref_seq)
        waveform = tx_objects["waveform"]
        ss_td_with_cp = tx_objects["ss_td_with_cp"]
        pdsch_idx = tx_objects["pdsch_idx"]

        channel_estimates_fd = []
        channel_mean_fd = []
        channel_impulse_response_td = []
        channel_full_spectrum_fd = []
        channel_impulse_response_td_single_pilot = []
        channel_full_spectrum_fd_single_pilot = []
        channel_estimates_fd_single_pilot = []
        channel_estimates_fd_raw_virtual = []
        channel_estimates_fd_aligned_virtual = []
        delay_profile_residual_gain_db = []
        virtual_pilot_variance_raw = []
        virtual_pilot_variance_aligned = []
        virtual_pilot_common_phases = []
        sync_indices = []
        ffo_estimates_hz = []
        rfo_estimates_hz = []
        timing_correlation_peaks = []
        single_pilot_snr_db = []
        virtual_pilot_snr_db = []
        virtual_pilot_snr_gain_db = []
        virtual_pilot_correction_phase_rad = []

        example_iq_rcv = None
        example_frame_rcv = None
        example_iq_rcv_single = None

        for capture_idx in range(NUM_CHANNEL_CAPTURES):
            print(
                f"[BW {bandwidth_mhz:>5} MHz] Capture "
                f"{capture_idx + 1}/{NUM_CHANNEL_CAPTURES}"
            )

            rx_objects = receive_and_process(
                usrp=usrp,
                params=params,
                waveform=waveform,
                ss_td_with_cp=ss_td_with_cp,
                pdsch_idx=pdsch_idx,
                known_ref_seq=known_ref_seq,
            )

            h_fd = rx_objects["h_fd"]
            h_mean = np.mean(h_fd, axis=(0, 1))
            h_single = np.mean(rx_objects["h_single_fd"], axis=(0, 1))
            h_delay, h_full_spectrum = channel_to_delay_response(
                h_mean,
                params["N"],
                params["FFT_SIZE"],
            )
            h_delay_single, h_full_spectrum_single = channel_to_delay_response(
                h_single,
                params["N"],
                params["FFT_SIZE"],
            )

            _, residual_single, _ = _delay_residual_power_db(
                h_delay_single,
                guard_taps=max(1, int(round(params["FFT_SIZE"] / 200))),
            )
            _, residual_avg, _ = _delay_residual_power_db(
                h_delay,
                guard_taps=max(1, int(round(params["FFT_SIZE"] / 200))),
            )
            residual_gain_db = 10.0 * np.log10((residual_single + 1e-15) / (residual_avg + 1e-15))

            channel_estimates_fd.append(h_fd.astype(np.complex64))
            channel_estimates_fd_single_pilot.append(rx_objects["h_single_fd"].astype(np.complex64))
            channel_estimates_fd_raw_virtual.append(rx_objects["h_fd_virtual_avg_raw"].astype(np.complex64))
            channel_estimates_fd_aligned_virtual.append(rx_objects["h_fd_virtual_avg_aligned"].astype(np.complex64))
            channel_mean_fd.append(h_mean.astype(np.complex64))
            channel_impulse_response_td.append(h_delay.astype(np.complex64))
            channel_full_spectrum_fd.append(h_full_spectrum.astype(np.complex64))
            channel_impulse_response_td_single_pilot.append(h_delay_single.astype(np.complex64))
            channel_full_spectrum_fd_single_pilot.append(h_full_spectrum_single.astype(np.complex64))
            delay_profile_residual_gain_db.append(float(residual_gain_db))
            virtual_pilot_variance_raw.append(rx_objects["virtual_pilot_variance_raw"].astype(np.float32))
            virtual_pilot_variance_aligned.append(rx_objects["virtual_pilot_variance_aligned"].astype(np.float32))
            virtual_pilot_common_phases.append(rx_objects["virtual_pilot_common_phases_rad"].astype(np.float32))
            single_pilot_snr_db.append(rx_objects["single_pilot_snr_db"])
            virtual_pilot_snr_db.append(rx_objects["virtual_pilot_snr_db"])
            virtual_pilot_snr_gain_db.append(rx_objects["virtual_pilot_snr_gain_db"])
            virtual_pilot_correction_phase_rad.append(
                rx_objects["virtual_pilot_correction_phase_rad"]
            )
            sync_indices.append(rx_objects["sync_idx"])
            ffo_estimates_hz.append(rx_objects["ffo_hz"])
            rfo_estimates_hz.append(rx_objects["rfo_hz_per_slot"])
            timing_correlation_peaks.append(rx_objects["timing_correlation_peak"])

            if capture_idx == 0:
                example_iq_rcv = rx_objects["iq_rcv"]
                example_frame_rcv = rx_objects["frame_rcv"]
                example_iq_rcv_single = rx_objects["iq_rcv_single"]

        results = {
            "channel_estimates_fd": np.stack(channel_estimates_fd, axis=0),
            "channel_estimates_fd_single_pilot": np.stack(channel_estimates_fd_single_pilot, axis=0),
            "channel_estimates_fd_raw_virtual": np.stack(channel_estimates_fd_raw_virtual, axis=0),
            "channel_estimates_fd_aligned_virtual": np.stack(channel_estimates_fd_aligned_virtual, axis=0),
            "channel_mean_fd": np.stack(channel_mean_fd, axis=0),
            "channel_impulse_response_td": np.stack(channel_impulse_response_td, axis=0),
            "channel_full_spectrum_fd": np.stack(channel_full_spectrum_fd, axis=0),
            "channel_impulse_response_td_single_pilot": np.stack(channel_impulse_response_td_single_pilot, axis=0),
            "channel_full_spectrum_fd_single_pilot": np.stack(channel_full_spectrum_fd_single_pilot, axis=0),
            "virtual_pilot_variance_raw": np.stack(virtual_pilot_variance_raw, axis=0),
            "virtual_pilot_variance_aligned": np.stack(virtual_pilot_variance_aligned, axis=0),
            "virtual_pilot_common_phases_rad": np.stack(virtual_pilot_common_phases, axis=0),
            "single_pilot_snr_db": np.asarray(single_pilot_snr_db, dtype=np.float32),
            "virtual_pilot_snr_db": np.asarray(virtual_pilot_snr_db, dtype=np.float32),
            "virtual_pilot_snr_gain_db": np.asarray(virtual_pilot_snr_gain_db, dtype=np.float32),
            "virtual_pilot_correction_phase_rad": np.asarray(
                virtual_pilot_correction_phase_rad, dtype=np.float32
            ),
            "delay_profile_residual_gain_db": np.asarray(delay_profile_residual_gain_db, dtype=np.float32),
            "sync_indices": np.asarray(sync_indices, dtype=np.int32),
            "ffo_estimates_hz": np.asarray(ffo_estimates_hz, dtype=np.float32),
            "rfo_estimates_hz": np.stack(rfo_estimates_hz, axis=0).astype(np.float32),
            "timing_correlation_peaks": np.asarray(timing_correlation_peaks, dtype=np.float32),
            "example_iq_rcv": example_iq_rcv.astype(np.complex64),
            "example_iq_rcv_single": example_iq_rcv_single.astype(np.complex64),
            "example_frame_rcv": example_frame_rcv.astype(np.complex64),
        }

        npz_path, json_path, fig_path, metadata = save_results(
            output_dir=OUTPUT_DIR,
            params=params,
            known_ref_seq=known_ref_seq,
            results=results,
        )

        plot_paths = plot_results(fig_path, params, results)

        print("Saved results:")
        print(f"  NPZ  : {npz_path}")
        print(f"  JSON : {json_path}")
        for key, path in plot_paths.items():
            print(f"  Plot ({key}): {path}")
        print("Important saved metadata:")
        for key in [
            "bandwidth_mhz",
            "subcarrier_spacing_hz",
            "num_active_subcarriers",
            "fft_size",
            "sampling_rate_hz",
            "frame_length_samples",
            "num_channel_captures",
            "saved_channel_tensor_shape",
            "saved_mean_channel_shape",
            "saved_delay_response_shape",
        ]:
            print(f"  {key}: {metadata[key]}")


if __name__ == "__main__":
    main()
