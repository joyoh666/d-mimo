"""Offline tests for the paper-compatible sounding transmitter."""

from __future__ import annotations

import unittest

import numpy as np

from USRP.modulate import modulate
from USRP.sounding_config import DEFAULT_SOUNDING_CONFIG
from USRP.sounding_transmitter import (
    DEFAULT_HARDWARE_CONFIG,
    build_sounding_period,
    build_transmitted_pilots,
    validate_sounding_period,
)


class SoundingTransmitterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DEFAULT_SOUNDING_CONFIG
        self.waveform = build_sounding_period(self.config)

    def test_period_shape_timing_and_scale(self) -> None:
        validate_sounding_period(self.waveform, self.config)
        self.assertEqual(
            self.waveform.shape,
            (self.config.num_tx_branches, self.config.csi_interval_samples),
        )
        self.assertEqual(self.waveform.dtype, np.complex64)
        self.assertAlmostEqual(
            float(np.max(np.abs(self.waveform))),
            self.config.digital_power_scale,
            places=6,
        )

    def test_zc_guard_and_padding_layout(self) -> None:
        zc_stop = self.config.zc_length_samples
        guard_stop = self.config.pilot_block_cp_start
        sync_branch = self.config.sync_tx_branch
        other_branches = np.arange(self.config.num_tx_branches) != sync_branch
        self.assertTrue(
            np.any(np.abs(self.waveform[sync_branch, :zc_stop]) > 0)
        )
        self.assertTrue(
            np.allclose(self.waveform[other_branches, :zc_stop], 0.0)
        )
        self.assertTrue(
            np.allclose(self.waveform[:, zc_stop:guard_stop], 0.0)
        )
        self.assertTrue(
            np.allclose(
                self.waveform[:, self.config.sounding_burst_samples :],
                0.0,
            )
        )

    def test_each_pilot_has_a_valid_cyclic_prefix(self) -> None:
        for repetition in range(self.config.pilot_repetitions):
            cp_start = self.config.pilot_symbol_cp_start(repetition)
            data_start = self.config.pilot_symbol_data_start(repetition)
            useful = self.waveform[
                :,
                data_start : data_start + self.config.fft_size,
            ]
            np.testing.assert_allclose(
                self.waveform[:, cp_start:data_start],
                useful[:, -self.config.cp_length_samples :],
                atol=1e-7,
            )

    def test_each_branch_occupies_only_its_assigned_bin(self) -> None:
        pilots = build_transmitted_pilots(self.config)
        for repetition in range(self.config.pilot_repetitions):
            start = self.config.pilot_symbol_data_start(repetition)
            useful = self.waveform[:, start : start + self.config.fft_size]
            bins = np.fft.fft(useful, axis=-1, norm="ortho")
            for branch, centered_bin in enumerate(
                self.config.pilot_centered_bins
            ):
                expected_bin = centered_bin % self.config.fft_size
                occupied = np.flatnonzero(np.abs(bins[branch]) > 1e-6)
                np.testing.assert_array_equal(occupied, [expected_bin])
                np.testing.assert_allclose(
                    bins[branch, expected_bin],
                    pilots[repetition, branch],
                    atol=2e-6,
                )

    def test_pilots_use_repository_modulate_mapping(self) -> None:
        rng = np.random.default_rng(self.config.pilot_seed)
        bits = rng.integers(
            0,
            2,
            size=(
                self.config.pilot_repetitions
                * self.config.num_tx_branches
                * 2
            ),
            dtype=np.uint8,
        )
        expected = modulate(bits, modulation_order=4).reshape(
            self.config.pilot_repetitions,
            self.config.num_tx_branches,
        )
        expected *= (
            np.sqrt(self.config.fft_size)
            * self.config.digital_power_scale
        )
        np.testing.assert_allclose(
            build_transmitted_pilots(self.config),
            expected.astype(np.complex64),
            atol=1e-7,
        )

    def test_device_arguments_preserve_original_ips(self) -> None:
        args = DEFAULT_HARDWARE_CONFIG.device_args(self.config)
        self.assertIn("addr0=192.168.110.2", args)
        self.assertIn("addr1=192.168.10.2", args)
        self.assertIn("addr2=192.168.100.2", args)
        self.assertIn("master_clock_rate=184320000", args)


if __name__ == "__main__":
    unittest.main()
