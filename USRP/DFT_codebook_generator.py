import numpy as np


def UPA_codebook_generator_DFT(
    Mx,
    My,
    Mz,
    oversampling_x=1,
    oversampling_y=1,
    oversampling_z=1,
    ant_spacing=0.5,
):
    """
    Generate an oversampled DFT codebook.

    Parameters
    ----------
    Mx, My, Mz : int
        Number of antenna elements on each axis.
    oversampling_x, oversampling_y, oversampling_z : int
        Oversampling factors on each axis.
    ant_spacing : float
        Antenna spacing normalized by wavelength.

        Note: This argument is retained for compatibility with the
        MATLAB function but is not used by the original DFT formula.

    Returns
    -------
    F_CB : ndarray, complex
        Codebook with shape
        (Mx*My*Mz,
         codebook_size_x*codebook_size_y*codebook_size_z).

    all_beams : ndarray, int
        Axis-wise beam indices with shape (num_beams, 3).
        Python's zero-based indexing is used.
    """

    # Number of codewords along each axis
    codebook_size_x = int(Mx * oversampling_x)
    codebook_size_y = int(My * oversampling_y)
    codebook_size_z = int(Mz * oversampling_z)

    # Antenna-element indices
    antx_idx = np.arange(Mx)
    anty_idx = np.arange(My)
    antz_idx = np.arange(Mz)

    # Quantized DFT spatial frequencies
    theta_qx = 2 * np.pi * np.arange(codebook_size_x) / codebook_size_x
    theta_qy = 2 * np.pi * np.arange(codebook_size_y) / codebook_size_y
    theta_qz = 2 * np.pi * np.arange(codebook_size_z) / codebook_size_z

    # Axis-wise DFT codebooks
    F_CBx = np.exp(
        -1j * antx_idx[:, None] * theta_qx[None, :]
    ) / np.sqrt(Mx)

    F_CBy = np.exp(
        -1j * anty_idx[:, None] * theta_qy[None, :]
    ) / np.sqrt(My)

    F_CBz = np.exp(
        -1j * antz_idx[:, None] * theta_qz[None, :]
    ) / np.sqrt(Mz)

    # Same Kronecker-product order as the MATLAB code
    F_CBxy = np.kron(F_CBy, F_CBx)
    F_CB = np.kron(F_CBz, F_CBxy)

    # Zero-based beam indices
    beams_x = np.arange(codebook_size_x)
    beams_y = np.arange(codebook_size_y)
    beams_z = np.arange(codebook_size_z)

    # x changes fastest, then y, then z
    Mxx_idx = np.tile(
        beams_x,
        codebook_size_y * codebook_size_z,
    )

    Myy_idx = np.tile(
        np.repeat(beams_y, codebook_size_x),
        codebook_size_z,
    )

    Mzz_idx = np.repeat(
        beams_z,
        codebook_size_x * codebook_size_y,
    )

    all_beams = np.column_stack(
        (Mxx_idx, Myy_idx, Mzz_idx)
    )

    return F_CB, all_beams