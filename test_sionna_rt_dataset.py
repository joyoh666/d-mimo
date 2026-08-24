from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from config import SoundingConfig
from sionna_rt_dataset import (
    add_pilot_noise,
    build_parser,
    generate_trajectory,
    main,
    validate_runtimes,
)


def test_trajectory_is_reproducible_and_bounded() -> None:
    arguments = dict(
        num_samples=60,
        samples_per_rt_update=20,
        sample_interval_s=0.005,
        bounds=(-0.5, 0.5, -0.4, 0.4),
        height_m=1.0,
        speed_mps=1.0,
        turn_std_deg=15.0,
        start_position=(0.0, 0.0, 1.0),
    )
    first = generate_trajectory(rng=np.random.default_rng(17), **arguments)
    second = generate_trajectory(rng=np.random.default_rng(17), **arguments)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.all((-0.5 <= first[0][:, 0]) & (first[0][:, 0] <= 0.5))
    assert np.all((-0.4 <= first[0][:, 1]) & (first[0][:, 1] <= 0.4))
    np.testing.assert_allclose(first[0][:, 2], 1.0)


def test_three_pilot_noise_is_averaged_in_complex_domain() -> None:
    truth = np.ones((1_000, 3, 2), dtype=np.complex64) * (1.0 + 1.0j)
    averaged, per_pilot, variance = add_pilot_noise(
        truth,
        repetitions=3,
        snr_db=20.0,
        rng=np.random.default_rng(23),
    )

    assert averaged.shape == truth.shape
    assert per_pilot.shape == (1_000, 3, 3, 2)
    np.testing.assert_allclose(averaged, np.mean(per_pilot, axis=1), atol=1e-7)
    np.testing.assert_allclose(variance, 0.02, rtol=1e-6)
    assert np.mean(np.abs(averaged - truth) ** 2) < np.mean(
        np.abs(per_pilot[:, 0] - truth) ** 2
    )


def test_pattern_and_scattering_sweep_expands_without_zero_duplicates() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tx-patterns",
            "iso,dipole",
            "--scattering-coefficients",
            "0,0.3",
            "--scattering-patterns",
            "lambertian,directive",
            "--xpd-coefficients",
            "0,0.2",
            "--samples-per-src",
            "1000",
        ]
    )
    cfg = SoundingConfig()
    runtimes = validate_runtimes(parser, args, cfg)

    # Per TX pattern: one deduplicated scattering=0 case, plus the
    # 2 scattering patterns x 2 XPD values for scattering=0.3.
    assert len(runtimes) == 2 * (1 + 2 * 2)
    assert len({runtime.variant_id for runtime in runtimes}) == len(runtimes)
    assert all(
        runtime.diffuse_reflection
        for runtime in runtimes
        if runtime.scattering_coefficient > 0.0
    )
    zero_scattering = [
        runtime
        for runtime in runtimes
        if runtime.scattering_coefficient == 0.0
    ]
    assert len(zero_scattering) == 2
    assert all(runtime.xpd_coefficient == 0.0 for runtime in zero_scattering)


def test_small_sionna_rt_episode_matches_collector_schema() -> None:
    with tempfile.TemporaryDirectory(prefix="d_mimo_sionna_test_") as directory:
        result = main(
            [
                "--scene",
                "empty",
                "--duration-s",
                "0.1",
                "--episodes",
                "1",
                "--geometry-update-ms",
                "100",
                "--max-depth",
                "0",
                "--samples-per-src",
                "100",
                "--max-num-paths-per-src",
                "1000",
                "--snr-db",
                "inf",
                "--speed-mps",
                "0.5",
                "--output-dir",
                directory,
                "--quiet",
            ]
        )
        assert result == 0
        path = next(Path(directory).rglob("*.npz"))
        assert path.parent.name == "tx-iso-V_rx-iso-V_sc-none-s0-xpd0"
        with np.load(path, allow_pickle=False) as saved:
            metadata_text = saved["metadata_json"].item()
            metadata = json.loads(metadata_text)
            assert saved["csi"].shape == (20, 3, 2)
            assert saved["per_pilot_csi"].shape == (20, 3, 3, 2)
            assert saved["predictor_tokens"].shape == (4, 3, 2, 15)
            assert saved["scheduler_segment_mean_gain"].shape == (1, 3)
            assert saved["csi_ground_truth"].shape == (20, 3, 2)
            assert saved["ue_position_m"].shape == (20, 3)
            assert saved["rt_path_count_per_ru"].shape == (1, 3)
            assert np.all(saved["valid"])
            np.testing.assert_allclose(
                saved["csi"],
                saved["csi_ground_truth"],
                atol=1e-8,
            )
            assert metadata["dataset_kind"] == "synthetic_sionna_rt"
            assert metadata["dataset_variant_id"] == path.parent.name
            assert metadata["scene"]["tx_pattern"] == "iso"
            assert metadata["scene"]["scattering_coefficient"] == 0.0
            assert metadata["episode_labels"]["trajectory_id"].startswith(
                "synthetic_rt_trajectory_"
            )
            assert metadata["source"]["normalize_delays"] is False
            assert metadata["noise"]["configured_per_pilot_snr_db"] == "+inf"
            assert "Infinity" not in metadata_text


if __name__ == "__main__":
    test_trajectory_is_reproducible_and_bounded()
    test_three_pilot_noise_is_averaged_in_complex_domain()
    test_pattern_and_scattering_sweep_expands_without_zero_duplicates()
    test_small_sionna_rt_episode_matches_collector_schema()
    print("All Sionna RT dataset tests passed.")
