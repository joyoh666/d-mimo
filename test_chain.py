from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from channel import apply_channel, frequency_response_at_bins, make_example_channels
from config import SoundingConfig
from receiver import receive_frame
from transmitter import build_transmit_frame, zadoff_chu


def test_paper_parameters() -> None:
    cfg = SoundingConfig()

    assert cfg.center_frequency_hz == 2.2e9
    assert cfg.nominal_bandwidth_hz == 1.4e6
    assert cfg.subcarrier_spacing_hz == 15e3
    assert cfg.nfft == 128
    assert cfg.pilot_period_s == 5e-3
    assert cfg.num_tx_branches == 6
    assert cfg.pilot_repetitions == 3
    assert cfg.sample_rate_hz == 1.92e6
    assert cfg.period_samples == 9600


def test_zc_has_constant_amplitude() -> None:
    zc = zadoff_chu(25, 139)
    np.testing.assert_allclose(np.abs(zc), 1.0, atol=1e-12)


def test_transmitter_repeats_three_identical_pilots() -> None:
    cfg = SoundingConfig()
    tx = build_transmit_frame(cfg)

    for branch in range(cfg.num_tx_branches):
        symbols = []
        for repetition in range(cfg.pilot_repetitions):
            start = cfg.pilot_symbol_cp_start(repetition)
            stop = start + cfg.pilot_symbol_samples
            symbols.append(tx.branch_samples[branch, start:stop])
        np.testing.assert_allclose(symbols[1], symbols[0], atol=1e-12)
        np.testing.assert_allclose(symbols[2], symbols[0], atol=1e-12)


def test_end_to_end_sounding_chain() -> None:
    cfg = SoundingConfig(snr_db=40.0)
    tx = build_transmit_frame(cfg)
    taps = make_example_channels(cfg)
    output = apply_channel(tx.branch_samples, taps, cfg)
    rx = receive_frame(output.received, tx.zc, tx.pilot_symbols, cfg)

    truth = frequency_response_at_bins(taps, cfg.pilot_bins, cfg.nfft)
    nmse = np.sum(np.abs(rx.channel_estimates - truth) ** 2) / np.sum(
        np.abs(truth) ** 2
    )

    assert abs(rx.sync_start - cfg.timing_offset_samples) <= 1
    assert abs(rx.cfo_estimate_hz - cfg.cfo_hz) < 30.0
    assert rx.received_grids.shape == (cfg.pilot_repetitions, cfg.nfft)
    assert rx.per_pilot_channel_estimates.shape == (
        cfg.pilot_repetitions,
        cfg.num_tx_branches,
    )
    assert 10.0 * np.log10(nmse) < -20.0


def test_three_pilot_average_reduces_mean_squared_error() -> None:
    cfg = SoundingConfig(snr_db=15.0, random_seed=19)
    tx = build_transmit_frame(cfg)
    taps = make_example_channels(cfg)
    output = apply_channel(tx.branch_samples, taps, cfg)
    rx = receive_frame(output.received, tx.zc, tx.pilot_symbols, cfg)
    truth = frequency_response_at_bins(taps, cfg.pilot_bins, cfg.nfft)

    individual_mse = np.mean(
        np.abs(rx.per_pilot_channel_estimates - truth[np.newaxis, :]) ** 2,
        axis=1,
    )
    averaged_mse = np.mean(np.abs(rx.channel_estimates - truth) ** 2)

    assert averaged_mse <= np.mean(individual_mse) + 1e-15


if __name__ == "__main__":
    test_paper_parameters()
    test_zc_has_constant_amplitude()
    test_transmitter_repeats_three_identical_pilots()
    test_end_to_end_sounding_chain()
    test_three_pilot_average_reduces_mean_squared_error()
    print("All sounding-chain tests passed.")
