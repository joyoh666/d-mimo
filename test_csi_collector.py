from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from channel import apply_channel, frequency_response_at_bins, make_example_channels
from config import SoundingConfig
from csi_collector import (
    estimate_episode_csi,
    make_model_views,
    make_time_indices,
    save_episode,
)
from transmitter import build_transmit_frame


def simulated_episode():
    cfg = SoundingConfig(snr_db=40.0)
    tx = build_transmit_frame(cfg)
    taps = make_example_channels(cfg)
    waveform_scale = 0.25
    channel_output = apply_channel(
        tx.branch_samples * waveform_scale,
        taps,
        cfg,
    )
    episode = estimate_episode_csi(
        channel_output.received,
        tx,
        cfg,
        num_frames=1,
        rx_start_time_s=0.0,
        tx_start_time_s=cfg.timing_offset_samples / cfg.sample_rate_hz,
        search_radius_samples=64,
        minimum_sync_metric=0.01,
        metadata={"episode_id": "simulation"},
        waveform_scale=waveform_scale,
    )
    truth = frequency_response_at_bins(taps, cfg.pilot_bins, cfg.nfft).reshape(3, 2)
    return cfg, episode, truth


def test_time_indices_match_paper_timescales() -> None:
    indices = make_time_indices(3_000)

    assert indices["predictor_block_index"][0] == 0
    assert indices["predictor_block_index"][4] == 0
    assert indices["predictor_block_index"][5] == 1
    assert indices["predictor_block_index"][-1] == 599
    assert indices["scheduler_segment_index"][19] == 0
    assert indices["scheduler_segment_index"][20] == 1
    assert indices["scheduler_segment_index"][-1] == 149


def test_model_views_follow_equation_9_and_equation_13() -> None:
    phase = np.linspace(-np.pi, np.pi, 20, endpoint=False, dtype=np.float32)
    csi = np.empty((20, 3, 2), dtype=np.complex64)
    for ru in range(3):
        for antenna in range(2):
            gain = np.float32(1.0 + ru + 0.25 * antenna)
            csi[:, ru, antenna] = gain * np.exp(1j * phase)
    views = make_model_views(csi, np.ones(20, dtype=bool))

    assert views["predictor_tokens"].shape == (4, 3, 2, 15)
    np.testing.assert_allclose(
        views["predictor_tokens"][0, 0, 0, :5],
        1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        views["predictor_tokens"][0, 0, 0, 5:10],
        np.cos(phase[:5]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        views["predictor_tokens"][0, 0, 0, 10:],
        np.sin(phase[:5]),
        atol=1e-6,
    )
    assert views["scheduler_segment_mean_gain"].shape == (1, 3)
    np.testing.assert_allclose(
        views["scheduler_segment_mean_gain"][0, 0],
        np.hypot(1.0, 1.25),
        atol=1e-6,
    )


def test_scaled_sounding_recovers_six_individual_channels() -> None:
    _, episode, truth = simulated_episode()

    assert episode.valid.tolist() == [True]
    assert episode.csi.shape == (1, 3, 2)
    assert episode.per_pilot_csi.shape == (1, 3, 3, 2)
    nmse = np.sum(np.abs(episode.csi[0] - truth) ** 2) / np.sum(
        np.abs(truth) ** 2
    )
    assert 10.0 * np.log10(nmse) < -20.0


def test_npz_is_self_describing_and_round_trips() -> None:
    _, episode, _ = simulated_episode()
    episode = replace(
        episode,
        extra_arrays={"ue_position_m": np.zeros((1, 3), dtype=np.float32)},
    )
    with tempfile.TemporaryDirectory(prefix="d_mimo_csi_test_") as directory:
        output_path = Path(directory) / "episode.npz"
        save_episode(output_path, episode)
        with np.load(output_path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["csi"], episode.csi)
            np.testing.assert_array_equal(saved["valid"], episode.valid)
            metadata = json.loads(saved["metadata_json"].item())
            assert metadata["episode_id"] == "simulation"
            assert saved["predictor_tokens"].shape == (0, 3, 2, 15)
            assert saved["ue_position_m"].shape == (1, 3)


if __name__ == "__main__":
    test_time_indices_match_paper_timescales()
    test_model_views_follow_equation_9_and_equation_13()
    test_scaled_sounding_recovers_six_individual_channels()
    test_npz_is_self_describing_and_round_trips()
    print("All CSI-collector tests passed.")
