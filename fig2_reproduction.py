"""Reproduce Fig. 2 from Wang et al., IEEE JSAC, 2013.

The script evaluates the four curves shown in Fig. 2 of
"Spectral Efficiency of Distributed MIMO Systems":

1. Exact Monte Carlo ergodic capacity from equation (11).
2. The ergodic-capacity lower bound in equation (30).
3. The high-SNR asymptotic approximation in equation (36).
4. The Schwartz-Yeh (SY) lognormal-sum approximation applied to
   the high-SNR lower bound in equation (35).

The cell radius D is normalized to one because only normalized distances
appear in the equations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import digamma, gammaln, polygamma


@dataclass(frozen=True)
class Figure2Config:
    """Parameters stated in the paper for Fig. 2."""

    mobile_antennas: int = 2  # M
    antennas_per_rau: int = 2  # L
    num_raus: int = 7  # N
    path_loss_exponent: float = 3.7  # alpha
    shadowing_std_db: float = 8.0  # sigma_sh
    snr_db: float = 20.0  # gamma in dB
    user_angle_rad: float = 0.0  # theta

    @property
    def snr_linear(self) -> float:
        return 10.0 ** (self.snr_db / 10.0)

    @property
    def lognormal_scale(self) -> float:
        return np.log(10.0) / 10.0


def rau_polar_coordinates(config: Figure2Config) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized RAU radii and angles specified below Fig. 2."""

    if config.num_raus != 7:
        raise ValueError("The paper's Fig. 2 geometry requires exactly seven RAUs.")

    ring_radius = (3.0 - np.sqrt(3.0)) / 2.0
    radii = np.array([0.0] + [ring_radius] * 6)
    angles = np.array(
        [
            0.0,
            np.pi / 6.0,
            np.pi / 2.0,
            5.0 * np.pi / 6.0,
            7.0 * np.pi / 6.0,
            3.0 * np.pi / 2.0,
            11.0 * np.pi / 6.0,
        ]
    )
    return radii, angles


def user_to_rau_distances(
    normalized_radii: np.ndarray, config: Figure2Config
) -> np.ndarray:
    """Evaluate equation (6) with D normalized to one."""

    rau_radii, rau_angles = rau_polar_coordinates(config)
    rho = np.asarray(normalized_radii, dtype=float)[:, None]
    squared_distances = (
        rho**2
        + rau_radii[None, :] ** 2
        - 2.0
        * rho
        * rau_radii[None, :]
        * np.cos(config.user_angle_rad - rau_angles[None, :])
    )
    distances = np.sqrt(np.maximum(squared_distances, 0.0))
    if np.any(distances == 0.0):
        raise ValueError("A user position coincides with an RAU; choose rho/D > 0.")
    return distances


def simulate_ergodic_capacity(
    distances: np.ndarray,
    config: Figure2Config,
    *,
    trials: int,
    seed: int,
    batch_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo evaluation of the exact mutual information in equation (11).

    The same random channel samples are reused across user positions. This common
    random-number strategy produces a smoother spatial curve without changing
    the expectation at any position.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    m = config.mobile_antennas
    ell = config.antennas_per_rau
    if m != 2:
        raise ValueError("The optimized Fig. 2 simulator currently requires M=2.")

    rng = np.random.default_rng(seed)
    path_gains = distances ** (-config.path_loss_exponent)
    snr_per_stream = config.snr_linear / m
    capacity_sum = np.zeros(distances.shape[0])
    capacity_square_sum = np.zeros(distances.shape[0])

    for first in range(0, trials, batch_size):
        current_batch = min(batch_size, trials - first)

        shadowing = np.exp(
            config.lognormal_scale
            * config.shadowing_std_db
            * rng.standard_normal((current_batch, config.num_raus))
        )
        rayleigh = (
            rng.standard_normal((current_batch, config.num_raus, ell, m))
            + 1j
            * rng.standard_normal((current_batch, config.num_raus, ell, m))
        ) / np.sqrt(2.0)

        # The three independent entries of H_w,n^H H_w,n for a 2x2 channel.
        gram_00 = np.sum(np.abs(rayleigh[..., 0]) ** 2, axis=2)
        gram_11 = np.sum(np.abs(rayleigh[..., 1]) ** 2, axis=2)
        gram_01 = np.sum(
            np.conjugate(rayleigh[..., 0]) * rayleigh[..., 1], axis=2
        )

        effective_gain = shadowing
        total_00 = np.einsum("rn,bn,bn->rb", path_gains, effective_gain, gram_00)
        total_11 = np.einsum("rn,bn,bn->rb", path_gains, effective_gain, gram_11)
        total_01 = np.einsum("rn,bn,bn->rb", path_gains, effective_gain, gram_01)

        determinant = (
            (1.0 + snr_per_stream * total_00)
            * (1.0 + snr_per_stream * total_11)
            - snr_per_stream**2 * np.abs(total_01) ** 2
        )
        determinant = np.maximum(determinant.real, np.finfo(float).tiny)
        capacities = np.log2(determinant)
        capacity_sum += np.sum(capacities, axis=1)
        capacity_square_sum += np.sum(capacities**2, axis=1)

    mean_capacity = capacity_sum / trials
    sample_variance = np.maximum(
        (capacity_square_sum - trials * mean_capacity**2) / max(trials - 1, 1),
        0.0,
    )
    standard_error = np.sqrt(sample_variance / trials)
    return mean_capacity, standard_error


def equation_30_lower_bound(
    distances: np.ndarray, config: Figure2Config
) -> np.ndarray:
    """Ergodic-capacity lower bound from equation (30)."""

    m = config.mobile_antennas
    ell = config.antennas_per_rau
    digamma_mean = sum(digamma(ell - index + 1) for index in range(1, m + 1)) / m
    path_sum = np.sum(distances ** (-config.path_loss_exponent), axis=1)
    return m * np.log2(
        1.0 + config.snr_linear / m * np.exp(digamma_mean) * path_sum
    )


def equation_36_asymptotic(
    distances: np.ndarray, config: Figure2Config
) -> np.ndarray:
    """High-SNR asymptotic mean from equation (36)."""

    m = config.mobile_antennas
    ell = config.antennas_per_rau
    gamma_ratio_log2 = sum(
        gammaln(ell - index + 1.0 / m + 1.0) - gammaln(ell - index + 1.0)
        for index in range(1, m + 1)
    ) / np.log(2.0)
    shadowing_term = (
        config.lognormal_scale**2
        * config.shadowing_std_db**2
        / (2.0 * np.log(2.0))
    )
    path_sum = np.sum(distances ** (-config.path_loss_exponent), axis=1)
    return m * (
        np.log2(config.snr_linear / m)
        + shadowing_term
        + gamma_ratio_log2
        + np.log2(path_sum)
    )


def combine_two_lognormals_sy(
    mean_1: float,
    variance_1: float,
    mean_2: float,
    variance_2: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return moments of log(exp(Y1) + exp(Y2)) for independent normals.

    This is the two-component step of Schwartz and Yeh's nested method. The
    original paper evaluates the moments with infinite series. Gauss-Hermite
    quadrature evaluates the same one-dimensional Gaussian expectations more
    directly and avoids cancellation in the series.
    """

    difference_mean = mean_2 - mean_1
    difference_variance = variance_1 + variance_2
    difference = difference_mean + np.sqrt(2.0 * difference_variance) * nodes
    softplus = np.logaddexp(0.0, difference)

    expected_softplus = np.sum(weights * softplus)
    expected_softplus_square = np.sum(weights * softplus**2)
    covariance_term = np.sum(
        weights * (difference - difference_mean) * softplus
    )

    combined_mean = mean_1 + expected_softplus
    combined_variance = (
        variance_1
        + expected_softplus_square
        - expected_softplus**2
        - 2.0 * variance_1 / difference_variance * covariance_term
    )
    return float(combined_mean), float(max(combined_variance, 0.0))


def schwartz_yeh_approximation(
    distances: np.ndarray,
    config: Figure2Config,
    *,
    quadrature_order: int = 80,
) -> np.ndarray:
    """Apply the Schwartz-Yeh approximation to equations (35) and (55)."""

    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least 8")

    m = config.mobile_antennas
    ell = config.antennas_per_rau
    fading_log_mean = sum(
        digamma(ell - index + 1) for index in range(1, m + 1)
    ) / m
    fading_log_variance = sum(
        polygamma(1, ell - index + 1) for index in range(1, m + 1)
    ) / m**2
    component_variance = (
        config.lognormal_scale**2 * config.shadowing_std_db**2
        + fading_log_variance
    )

    nodes, weights = hermgauss(quadrature_order)
    weights = weights / np.sqrt(np.pi)
    estimates = np.empty(distances.shape[0])

    for row, position_distances in enumerate(distances):
        component_means = (
            config.path_loss_exponent * np.log(1.0 / position_distances)
            + fading_log_mean
        )
        combined_mean = float(component_means[0])
        combined_variance = float(component_variance)

        # Preserve the RAU order printed in the paper. Schwartz and Yeh report
        # that permutations have little effect on the resulting mean.
        for component_mean in component_means[1:]:
            combined_mean, combined_variance = combine_two_lognormals_sy(
                combined_mean,
                combined_variance,
                float(component_mean),
                float(component_variance),
                nodes,
                weights,
            )

        estimates[row] = (
            m * np.log2(config.snr_linear / m)
            + m / np.log(2.0) * combined_mean
        )

    return estimates


def save_results_csv(
    output_path: Path,
    normalized_radii: np.ndarray,
    simulation: np.ndarray,
    simulation_standard_error: np.ndarray,
    lower_bound: np.ndarray,
    asymptotic: np.ndarray,
    schwartz_yeh: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "rho_over_D",
                "simulation_bit_s_hz",
                "simulation_standard_error",
                "lower_bound_eq30_bit_s_hz",
                "asymptotic_eq36_bit_s_hz",
                "schwartz_yeh_bit_s_hz",
            ]
        )
        writer.writerows(
            zip(
                normalized_radii,
                simulation,
                simulation_standard_error,
                lower_bound,
                asymptotic,
                schwartz_yeh,
            )
        )


def plot_figure_2(
    output_path: Path,
    normalized_radii: np.ndarray,
    simulation: np.ndarray,
    lower_bound: np.ndarray,
    asymptotic: np.ndarray,
    schwartz_yeh: np.ndarray,
    *,
    show: bool,
) -> None:
    """Create a plot styled after the paper's Fig. 2."""

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    marker_stride = max(1, len(normalized_radii) // 25)

    axis.plot(
        normalized_radii,
        simulation,
        color="blue",
        linewidth=1.8,
        label="Ergodic capacity, simulation",
    )
    axis.plot(
        normalized_radii,
        lower_bound,
        color="red",
        linewidth=1.5,
        marker="o",
        markerfacecolor="none",
        markeredgewidth=1.5,
        markevery=marker_stride,
        label="Lower bound",
    )
    axis.plot(
        normalized_radii,
        asymptotic,
        color="red",
        linewidth=1.5,
        marker="+",
        markersize=7,
        markevery=marker_stride,
        label="Asymptotic approximation",
    )
    axis.plot(
        normalized_radii,
        schwartz_yeh,
        color="magenta",
        linewidth=1.5,
        marker="x",
        markersize=5,
        markevery=marker_stride,
        label="Approximation (SY's method)",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(18.0, 53.0)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_yticks(np.arange(20.0, 51.0, 5.0))
    axis.set_xlabel(r"$\rho/D$")
    axis.set_ylabel("Ergodic capacity (bit/s/Hz)")
    axis.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    axis.tick_params(direction="in", top=True, right=True)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=100_000,
        help="Monte Carlo realizations per user position (default: 100000)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=50,
        help="Number of rho/D samples from 0.02 to 1.0 (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260811, help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figures/fig2_reproduction.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("output/figures/fig2_reproduction.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 2:
        raise ValueError("points must be at least 2")

    config = Figure2Config()
    normalized_radii = np.linspace(0.02, 1.0, args.points)
    distances = user_to_rau_distances(normalized_radii, config)

    simulation, standard_error = simulate_ergodic_capacity(
        distances, config, trials=args.trials, seed=args.seed
    )
    lower_bound = equation_30_lower_bound(distances, config)
    asymptotic = equation_36_asymptotic(distances, config)
    schwartz_yeh = schwartz_yeh_approximation(distances, config)

    plot_figure_2(
        args.output,
        normalized_radii,
        simulation,
        lower_bound,
        asymptotic,
        schwartz_yeh,
        show=args.show,
    )
    save_results_csv(
        args.csv_output,
        normalized_radii,
        simulation,
        standard_error,
        lower_bound,
        asymptotic,
        schwartz_yeh,
    )

    print(f"Saved figure: {args.output.resolve()}")
    print(f"Saved data:   {args.csv_output.resolve()}")
    print(f"Maximum Monte Carlo standard error: {standard_error.max():.4f} bit/s/Hz")


if __name__ == "__main__":
    main()
