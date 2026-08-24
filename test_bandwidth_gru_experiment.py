"""Unit tests for the standalone bandwidth/GRU experiment."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from bandwidth_gru_experiment import (
    ExperimentConfig,
    PILOT_FRACTIONS,
    PerLinkTokenDataset,
    SimpleChannelGRU,
    complex_to_tokens,
    evaluate,
    rollout,
    tokens_to_complex,
)


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        bandwidths_mhz=(1.4, 5.0, 9.0, 15.0),
        carrier_frequency_hz=2.2e9,
        delay_spread_s=30e-9,
        speed_mps=0.5,
        snr_db=30.0,
        windows=10,
        samples_per_window=40,
        samples_per_step=5,
        csi_period_s=5e-3,
        tdl_batch_size=5,
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
        patience=1,
        min_epochs=1,
    )


class BandwidthExperimentTests(unittest.TestCase):
    def test_token_round_trip(self) -> None:
        torch.manual_seed(1)
        h = torch.complex(torch.randn(4, 5), torch.randn(4, 5))
        token = complex_to_tokens(h, gain_scale=20.0)
        restored = tokens_to_complex(token, taps=5, gain_scale=20.0)
        self.assertTrue(torch.allclose(h, restored, atol=2e-6, rtol=2e-6))

    def test_dataset_and_rollout_shapes(self) -> None:
        cfg = tiny_config()
        rng = np.random.default_rng(2)
        x = (
            rng.standard_normal((cfg.windows, cfg.samples_per_window, len(PILOT_FRACTIONS)))
            + 1j
            * rng.standard_normal((cfg.windows, cfg.samples_per_window, len(PILOT_FRACTIONS)))
        ).astype(np.complex64)
        dataset = PerLinkTokenDataset(
            x, np.arange(8), cfg.samples_per_step, cfg.gain_scale, augment_phase=False
        )
        self.assertEqual(len(dataset), 8 * len(PILOT_FRACTIONS))
        seq = torch.stack((dataset[0], dataset[1]))
        self.assertEqual(tuple(seq.shape), (2, 8, 15))
        model = SimpleChannelGRU(cfg)
        pred = rollout(model, seq, t0=1, pred_steps=cfg.pred_steps)
        self.assertEqual(tuple(pred.shape), (2, 5, 15))
        gain = pred[..., : cfg.samples_per_step]
        phase_norm = torch.sqrt(
            pred[..., cfg.samples_per_step : 2 * cfg.samples_per_step].square()
            + pred[..., 2 * cfg.samples_per_step :].square()
        )
        self.assertTrue(bool(torch.all(gain >= 0)))
        self.assertTrue(bool(torch.all(phase_norm > 0.99)))
        self.assertTrue(bool(torch.all(phase_norm <= 1.0)))


if __name__ == "__main__":
    unittest.main()
