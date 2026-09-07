"""Reusable DFT beam sweeping for the prototype's 4x1 ULA.

The four DFT codewords are stored as columns: ``DFT_CODEBOOK[:, k]`` is
beam ``k``.  A hardware integration only needs to provide two callbacks:

``apply_beam(beam_index, weights)``
    Programs the phased array with four complex, constant-modulus weights.

``measure_response(beam_index)``
    Returns one pilot-normalized complex response, ``h^H d_k``.

Keeping the complex response is important.  Its magnitude is sufficient for
beam selection, while its conjugate is the DFT coefficient needed to
reconstruct the spatial-domain channel under the convention above.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


NUM_ULA_ELEMENTS = 4
NUM_DFT_BEAMS = NUM_ULA_ELEMENTS

ComplexArray = NDArray[np.complex64]
FloatArray = NDArray[np.float32]

ApplyBeam = Callable[[int, ComplexArray], None]
MeasureResponse = Callable[[int], complex]


def generate_4x1_ula_dft_codebook() -> ComplexArray:
    """Return the non-oversampled, unitary 4x4 DFT codebook.

    Rows correspond to antenna elements and columns correspond to beams.  The
    element weights follow

        d_k[n] = exp(-j 2 pi n k / 4) / sqrt(4).

    Each codeword therefore has unit norm and constant element magnitude 1/2.
    """

    element_indices = np.arange(NUM_ULA_ELEMENTS, dtype=np.float64)
    beam_indices = np.arange(NUM_DFT_BEAMS, dtype=np.float64)
    phases = -2j * np.pi * np.outer(element_indices, beam_indices)
    codebook = np.exp(phases / NUM_DFT_BEAMS) / np.sqrt(NUM_ULA_ELEMENTS)
    return np.asarray(codebook, dtype=np.complex64)


DFT_CODEBOOK = generate_4x1_ula_dft_codebook()


def _validate_codebook(codebook: ArrayLike) -> ComplexArray:
    matrix = np.asarray(codebook, dtype=np.complex64)
    expected_shape = (NUM_ULA_ELEMENTS, NUM_DFT_BEAMS)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"the 4x1 ULA codebook must have shape {expected_shape}, "
            f"got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("the codebook contains NaN or infinite values")
    gram = matrix.conj().T @ matrix
    if not np.allclose(gram, np.eye(NUM_DFT_BEAMS), atol=1e-6):
        raise ValueError("the codebook must contain four orthonormal beams")
    return matrix


def channel_to_dft(
    channel: ArrayLike,
    codebook: ArrayLike = DFT_CODEBOOK,
) -> ComplexArray:
    """Project a four-element spatial channel onto the DFT codebook."""

    matrix = _validate_codebook(codebook)
    spatial_channel = np.asarray(channel, dtype=np.complex64)
    if spatial_channel.shape != (NUM_ULA_ELEMENTS,):
        raise ValueError(
            "channel must contain one complex value for each of the four "
            f"ULA elements, got shape {spatial_channel.shape}"
        )
    if not np.all(np.isfinite(spatial_channel)):
        raise ValueError("channel contains NaN or infinite values")
    return np.asarray(matrix.conj().T @ spatial_channel, dtype=np.complex64)


def dft_to_channel(
    dft_coefficients: ArrayLike,
    codebook: ArrayLike = DFT_CODEBOOK,
) -> ComplexArray:
    """Reconstruct the spatial channel from all four complex DFT coefficients."""

    matrix = _validate_codebook(codebook)
    coefficients = np.asarray(dft_coefficients, dtype=np.complex64)
    if coefficients.shape != (NUM_DFT_BEAMS,):
        raise ValueError(
            f"dft_coefficients must have shape ({NUM_DFT_BEAMS},), "
            f"got {coefficients.shape}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("DFT coefficients contain NaN or infinite values")
    return np.asarray(matrix @ coefficients, dtype=np.complex64)


@dataclass(frozen=True, slots=True)
class BeamSweepResult:
    """Complex measurements and powers from one complete four-beam sweep."""

    beam_indices: NDArray[np.int64]
    responses: ComplexArray
    codebook: ComplexArray

    @property
    def mean_responses(self) -> ComplexArray:
        """Complex-average repeated measurements for every beam."""

        return np.asarray(np.mean(self.responses, axis=1), dtype=np.complex64)

    @property
    def mean_powers(self) -> FloatArray:
        """Average ``|h^H d_k|^2`` over repetitions for every beam."""

        return np.asarray(
            np.mean(np.abs(self.responses) ** 2, axis=1),
            dtype=np.float32,
        )

    @property
    def best_beam_index(self) -> int:
        """Index of the beam with the largest average measured power."""

        return int(self.beam_indices[int(np.argmax(self.mean_powers))])

    @property
    def best_beam_weights(self) -> ComplexArray:
        """Four phased-array weights for the strongest measured beam."""

        return self.codebook[:, self.best_beam_index].copy()

    @property
    def dft_coefficients(self) -> ComplexArray:
        """Return ``D^H h`` inferred from measurements of ``h^H d_k``.

        This property assumes the channel remains constant for the complete
        sweep and that ``measure_response`` removes the transmitted pilot.
        """

        return np.asarray(self.mean_responses.conj(), dtype=np.complex64)

    @property
    def reconstructed_channel(self) -> ComplexArray:
        """Return the four-element channel reconstructed from the full sweep."""

        return dft_to_channel(self.dft_coefficients, self.codebook)


def sweep_4x1_ula_dft_beams(
    apply_beam: ApplyBeam,
    measure_response: MeasureResponse,
    *,
    repetitions: int = 1,
    settle_time_s: float = 0.0,
    beam_order: Sequence[int] | None = None,
    codebook: ArrayLike = DFT_CODEBOOK,
) -> BeamSweepResult:
    """Apply and measure every beam in the 4x1 ULA DFT codebook.

    Parameters
    ----------
    apply_beam:
        Callback receiving ``(beam_index, weights)``.  Use it to program the
        phased-array hardware.  ``weights`` has shape ``(4,)``.
    measure_response:
        Callback returning one finite, pilot-normalized complex value using
        the convention ``h^H d_k``.
    repetitions:
        Number of measurements collected after each beam is applied.
    settle_time_s:
        Optional delay after programming a new beam.
    beam_order:
        Optional permutation of ``(0, 1, 2, 3)``.  The returned arrays are
        reordered back into ascending beam-index order.
    codebook:
        A 4x4 unitary codebook whose columns are phased-array weights.
    """

    if not callable(apply_beam):
        raise TypeError("apply_beam must be callable")
    if not callable(measure_response):
        raise TypeError("measure_response must be callable")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise TypeError("repetitions must be an integer")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if settle_time_s < 0:
        raise ValueError("settle_time_s must be non-negative")

    matrix = _validate_codebook(codebook)
    order = (
        tuple(range(NUM_DFT_BEAMS))
        if beam_order is None
        else tuple(int(index) for index in beam_order)
    )
    expected_order = tuple(range(NUM_DFT_BEAMS))
    if len(order) != NUM_DFT_BEAMS or set(order) != set(expected_order):
        raise ValueError(
            f"beam_order must be a permutation of {expected_order}, got {order}"
        )

    responses = np.empty(
        (NUM_DFT_BEAMS, repetitions),
        dtype=np.complex64,
    )
    for beam_index in order:
        weights = matrix[:, beam_index].copy()
        apply_beam(beam_index, weights)
        if settle_time_s:
            time.sleep(settle_time_s)

        for repetition in range(repetitions):
            response = complex(measure_response(beam_index))
            if not np.isfinite(response.real) or not np.isfinite(response.imag):
                raise ValueError(
                    f"non-finite response for beam {beam_index}, "
                    f"repetition {repetition}"
                )
            responses[beam_index, repetition] = response

    return BeamSweepResult(
        beam_indices=np.arange(NUM_DFT_BEAMS, dtype=np.int64),
        responses=responses,
        codebook=matrix.copy(),
    )


__all__ = [
    "BeamSweepResult",
    "DFT_CODEBOOK",
    "NUM_DFT_BEAMS",
    "NUM_ULA_ELEMENTS",
    "channel_to_dft",
    "dft_to_channel",
    "generate_4x1_ula_dft_codebook",
    "sweep_4x1_ula_dft_beams",
]
