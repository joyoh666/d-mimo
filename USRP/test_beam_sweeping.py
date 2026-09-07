"""Tests for the reusable 4x1 ULA DFT beam sweep."""

from __future__ import annotations

import unittest

import numpy as np

from USRP.DFTsweeping.beam_sweeping import (
    DFT_CODEBOOK,
    channel_to_dft,
    dft_to_channel,
    sweep_4x1_ula_dft_beams,
)


class BeamSweepingTests(unittest.TestCase):
    def test_default_codebook_is_unitary_and_constant_modulus(self) -> None:
        self.assertEqual(DFT_CODEBOOK.shape, (4, 4))
        np.testing.assert_allclose(
            DFT_CODEBOOK.conj().T @ DFT_CODEBOOK,
            np.eye(4),
            atol=1e-6,
        )
        np.testing.assert_allclose(np.abs(DFT_CODEBOOK), 0.5, atol=1e-7)

    def test_complex_dft_round_trip(self) -> None:
        channel = np.array(
            [1.0 + 0.2j, -0.4 + 0.7j, 0.1 - 0.3j, 0.8 + 0.9j],
            dtype=np.complex64,
        )
        recovered = dft_to_channel(channel_to_dft(channel))
        np.testing.assert_allclose(recovered, channel, atol=2e-7)

    def test_sweep_preserves_complex_coefficients_and_selects_best_beam(self) -> None:
        coefficients = np.array(
            [0.1 + 0.1j, 0.2 - 0.3j, 0.4 + 0.8j, -0.2 + 0.1j],
            dtype=np.complex64,
        )
        channel = dft_to_channel(coefficients)
        applied: list[tuple[int, np.ndarray]] = []

        def apply_beam(beam_index: int, weights: np.ndarray) -> None:
            applied.append((beam_index, weights.copy()))

        def measure_response(beam_index: int) -> complex:
            weights = DFT_CODEBOOK[:, beam_index]
            return complex(np.vdot(channel, weights))

        result = sweep_4x1_ula_dft_beams(
            apply_beam,
            measure_response,
            repetitions=2,
            beam_order=(2, 0, 3, 1),
        )

        self.assertEqual([item[0] for item in applied], [2, 0, 3, 1])
        self.assertEqual(result.responses.shape, (4, 2))
        self.assertEqual(result.best_beam_index, 2)
        np.testing.assert_allclose(
            result.dft_coefficients,
            coefficients,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            result.reconstructed_channel,
            channel,
            atol=2e-7,
        )

    def test_invalid_beam_order_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sweep_4x1_ula_dft_beams(
                lambda _index, _weights: None,
                lambda _index: 0j,
                beam_order=(0, 1, 1, 3),
            )


if __name__ == "__main__":
    unittest.main()
