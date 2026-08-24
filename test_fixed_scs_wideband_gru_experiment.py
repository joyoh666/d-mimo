"""Tests for the fixed-SCS full-band GRU experiment."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fixed_scs_wideband_gru_experiment import (
    WidebandChannelGRU,
    WidebandConfig,
    WidebandTokenDataset,
    centered_frequency_grid,
    complex_to_tokens,
    rollout,
    tokens_to_complex,
)


def tiny_config() -> WidebandConfig:
    return WidebandConfig(
        bandwidths_mhz=(1.4, 5.0, 9.0, 15.0),
        subcarrier_spacing_hz=15e3,
        carrier_frequency_hz=2.2e9,
        delay_spread_s=30e-9,
        speed_mps=0.5,
        snr_db=30.0,
        channel_gain=0.05,
        windows=10,
        samples_per_window=40,
        samples_per_step=5,
        csi_period_s=5e-3,
        tdl_batch_size=5,
        num_links=6,
        train_fraction=0.8,
        val_fraction=0.1,
        seed=123,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        gain_scale=20.0,
        eps_norm=1e-6,
        pred_steps=5,
        loss_gamma=0.5,
        max_age_feature=32,
        batch_size=4,
        epochs=1,
        learning_rate=2e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        eval_every_epochs=1,
        patience_evals=1,
        min_epochs=1,
        benchmark_repeats=2,
    )


class FixedScsWidebandTests(unittest.TestCase):
    def test_subcarrier_counts_and_spacing(self) -> None:
        cfg = tiny_config()
        self.assertEqual(
            [cfg.num_subcarriers(bw) for bw in cfg.bandwidths_mhz],
            [93, 333, 600, 1000],
        )
        grid = centered_frequency_grid(600, cfg.subcarrier_spacing_hz, torch.device("cpu"))
        self.assertTrue(
            torch.allclose(
                grid[1:] - grid[:-1],
                torch.full((599,), cfg.subcarrier_spacing_hz, dtype=torch.float64),
            )
        )

    def test_token_round_trip_and_rollout_shape(self) -> None:
        cfg = tiny_config()
        num_subcarriers = 7
        rng = np.random.default_rng(1)
        array = (
            rng.standard_normal((10, 40, 6, num_subcarriers))
            + 1j * rng.standard_normal((10, 40, 6, num_subcarriers))
        ).astype(np.complex64)
        dataset = WidebandTokenDataset(
            array, np.arange(8), cfg, augment_phase=False
        )
        sequence = torch.stack((dataset[0], dataset[1]))
        token_dim = 3 * cfg.samples_per_step * num_subcarriers
        self.assertEqual(tuple(sequence.shape), (2, 8, token_dim))
        model = WidebandChannelGRU(token_dim, cfg)
        prediction = rollout(model, sequence, t0=1, pred_steps=cfg.pred_steps)
        self.assertEqual(tuple(prediction.shape), (2, 5, token_dim))

        channel = torch.complex(torch.randn(3, 35), torch.randn(3, 35))
        tokens = complex_to_tokens(channel, cfg.gain_scale)
        restored = tokens_to_complex(tokens, 35, cfg.gain_scale)
        self.assertTrue(torch.allclose(channel, restored, atol=2e-6, rtol=2e-6))


if __name__ == "__main__":
    unittest.main()
