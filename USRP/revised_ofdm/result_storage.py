"""Persistence and metadata generation for OFDM channel-capture results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

if __package__:  # Support package-style imports.
    from .capture_pipeline import CaptureResults
    from .config import DEFAULT_CONFIG, OFDMConfig
else:  # Support direct execution from the USRP directory.
    from USRP.test.revised_ofdm.capture_pipeline import CaptureResults
    from USRP.test.revised_ofdm.config import DEFAULT_CONFIG, OFDMConfig


Metadata = dict[str, Any]


def build_metadata(
    config: OFDMConfig,
    results: CaptureResults,
    *,
    timestamp: str,
) -> Metadata:
    """Build the metadata dictionary emitted by the original program."""
    return {
        "timestamp": timestamp,
        "bandwidth_mhz": config.BANDWIDTH,
        "subcarrier_spacing_hz": config.DELTA_F,
        "num_active_subcarriers": config.N,
        "fft_size": config.FFT_SIZE,
        "sampling_rate_hz": config.sampling_rate,
        "carrier_frequency_hz": config.carrier_frequency,
        "tx_gain_db": config.Tx_gain,
        "rx_gain_db": config.Rx_gain,
        "num_symbols_per_slot": config.num_symbols_per_slot,
        "num_slots_per_subframe": config.num_slot_per_subframe,
        "num_subframes_per_frame": config.num_subframe_per_frame,
        "num_symbols_per_frame": config.num_symbols_frame,
        "normal_cp_length_samples": config.normal_CP_length,
        "first_cp_length_samples": config.first_CP_length,
        "slot_length_samples": config.slot_length,
        "frame_length_samples": config.frame_length,
        "num_channel_captures": int(results["channel_estimates_fd"].shape[0]),
        "saved_channel_tensor_shape": list(
            results["channel_estimates_fd"].shape
        ),
        "saved_mean_channel_shape": list(results["channel_mean_fd"].shape),
        "saved_delay_response_shape": list(
            results["channel_impulse_response_td"].shape
        ),
        "saved_single_pilot_delay_response_shape": list(
            results["channel_impulse_response_td_single_pilot"].shape
        ),
        "samples_per_channel_vector": config.N,
        "samples_per_delay_response": config.FFT_SIZE,
        "reference_sequence_type": "seeded_unit_power_qpsk",
        # Keep the original metadata value.  The actual per-bandwidth seed is
        # reproducible as this base value + int(10 * bandwidth_mhz).
        "reference_sequence_seed": config.REFERENCE_SEQUENCE_SEED,
        "virtual_pilot_count": config.NUM_VIRTUAL_PILOTS,
        "virtual_pilot_positions": config.get_virtual_pilot_positions(),
        "phase_align_virtual_pilots": bool(
            config.PHASE_ALIGN_VIRTUAL_PILOTS
        ),
        "expected_channel_estimation_snr_gain_db": float(
            10 * np.log10(config.NUM_VIRTUAL_PILOTS)
        ),
        "achieved_channel_estimation_snr_gain_db_mean": float(
            np.mean(results["virtual_pilot_snr_gain_db"])
        ),
        "mean_single_to_averaged_residual_delay_gain_db": float(
            np.mean(results["delay_profile_residual_gain_db"])
        ),
        "modulation_order": config.modulation_order,
        "digital_power_scale": config.POWER,
    }


def save_results(
    output_dir: str | Path,
    config: OFDMConfig,
    known_ref_seq: np.ndarray,
    results: CaptureResults,
    *,
    timestamp: str | None = None,
) -> tuple[str, str, str, Metadata]:
    """Save NPZ/JSON results and reserve the original summary-plot path.

    The NPZ field names and metadata keys intentionally match
    ``revised_ofdm_channel_capture_virtual7_timeavg.py`` so existing analysis
    code can consume either output without changes.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    bandwidth_tag = f"BW_{str(config.BANDWIDTH).replace('.', 'p')}MHz"
    npz_path = destination / f"channel_dataset_{bandwidth_tag}_{timestamp}.npz"
    json_path = (
        destination
        / f"channel_dataset_{bandwidth_tag}_{timestamp}_metadata.json"
    )
    figure_path = destination / f"channel_plots_{bandwidth_tag}_{timestamp}.png"

    metadata = build_metadata(config, results, timestamp=timestamp)
    np.savez_compressed(
        npz_path,
        channel_estimates_fd=results["channel_estimates_fd"],
        channel_estimates_fd_single_pilot=(
            results["channel_estimates_fd_single_pilot"]
        ),
        channel_estimates_fd_raw_virtual=(
            results["channel_estimates_fd_raw_virtual"]
        ),
        channel_estimates_fd_aligned_virtual=(
            results["channel_estimates_fd_aligned_virtual"]
        ),
        channel_mean_fd=results["channel_mean_fd"],
        virtual_pilot_variance_raw=results["virtual_pilot_variance_raw"],
        virtual_pilot_variance_aligned=(
            results["virtual_pilot_variance_aligned"]
        ),
        virtual_pilot_common_phases_rad=(
            results["virtual_pilot_common_phases_rad"]
        ),
        channel_impulse_response_td=(
            results["channel_impulse_response_td"]
        ),
        channel_impulse_response_td_single_pilot=(
            results["channel_impulse_response_td_single_pilot"]
        ),
        channel_full_spectrum_fd_single_pilot=(
            results["channel_full_spectrum_fd_single_pilot"]
        ),
        channel_full_spectrum_fd=results["channel_full_spectrum_fd"],
        known_reference_sequence=np.asarray(
            known_ref_seq,
            dtype=np.complex64,
        ),
        sync_indices=results["sync_indices"],
        ffo_estimates_hz=results["ffo_estimates_hz"],
        rfo_estimates_hz=results["rfo_estimates_hz"],
        timing_correlation_peaks=results["timing_correlation_peaks"],
        single_pilot_snr_db=results["single_pilot_snr_db"],
        virtual_pilot_snr_db=results["virtual_pilot_snr_db"],
        virtual_pilot_snr_gain_db=results["virtual_pilot_snr_gain_db"],
        delay_profile_residual_gain_db=(
            results["delay_profile_residual_gain_db"]
        ),
        virtual_pilot_correction_phase_rad=(
            results["virtual_pilot_correction_phase_rad"]
        ),
        example_rx_constellation=results["example_iq_rcv"],
        example_rx_frame=results["example_frame_rcv"],
        metadata_json=json.dumps(metadata),
    )
    with json_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return (
        str(npz_path),
        str(json_path),
        str(figure_path),
        metadata,
    )


def save_default_results(
    known_ref_seq: np.ndarray,
    results: CaptureResults,
    config: OFDMConfig = DEFAULT_CONFIG,
) -> tuple[str, str, str, Metadata]:
    """Save results into ``config.OUTPUT_DIR`` using the default naming."""
    return save_results(
        config.OUTPUT_DIR,
        config,
        known_ref_seq,
        results,
    )
