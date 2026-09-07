"""Modular entry point for the virtual-pilot OFDM channel capture.

This program reproduces the execution flow of
``revised_ofdm_channel_capture_virtual7_timeavg.py`` using the split modules:
configuration, frame generation, USRP I/O, synchronization, receiver DSP,
capture aggregation, result persistence, and plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np

if __package__:  # Support package-style imports.
    from .capture_pipeline import (
        CaptureFunction,
        CaptureResults,
        collect_channel_results,
    )
    from .config import DEFAULT_CONFIG, OFDMConfig, iter_capture_configs
    from .ofdm_frame import build_frame_from_config
    from .result_plotting import PlotPaths, plot_results
    from .result_storage import Metadata, save_results
else:  # Support direct execution from the USRP directory.
    from USRP.test.revised_ofdm.capture_pipeline import (
        CaptureFunction,
        CaptureResults,
        collect_channel_results,
    )
    from USRP.test.revised_ofdm.config import DEFAULT_CONFIG, OFDMConfig, iter_capture_configs
    from USRP.test.revised_ofdm.ofdm_frame import build_frame_from_config
    from USRP.test.revised_ofdm.result_plotting import PlotPaths, plot_results
    from USRP.test.revised_ofdm.result_storage import Metadata, save_results


class ExperimentArtifacts(TypedDict):
    """Files and in-memory data produced for one bandwidth."""

    config: OFDMConfig
    results: CaptureResults
    npz_path: str
    json_path: str
    plot_paths: PlotPaths
    metadata: Metadata


def _initialize_usrp(config: OFDMConfig) -> Any:
    """Import UHD only for a real hardware run."""
    if __package__:
        from .usrp_capture import initialize_usrp
    else:
        from USRP.test.revised_ofdm.usrp_capture import initialize_usrp

    return initialize_usrp(config)


def _print_saved_summary(
    npz_path: str,
    json_path: str,
    plot_paths: PlotPaths,
    metadata: Metadata,
) -> None:
    print("Saved results:")
    print(f"  NPZ  : {npz_path}")
    print(f"  JSON : {json_path}")
    for plot_name, plot_path in plot_paths.items():
        print(f"  Plot ({plot_name}): {plot_path}")
    print("Important saved metadata:")
    for key in (
        "bandwidth_mhz",
        "subcarrier_spacing_hz",
        "num_active_subcarriers",
        "fft_size",
        "sampling_rate_hz",
        "frame_length_samples",
        "num_channel_captures",
        "saved_channel_tensor_shape",
        "saved_mean_channel_shape",
        "saved_delay_response_shape",
    ):
        print(f"  {key}: {metadata[key]}")


def run_capture_experiment(
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    usrp: Any | None = None,
    capture_function: CaptureFunction | None = None,
    show_plots: bool = True,
) -> list[ExperimentArtifacts]:
    """Run every configured bandwidth and return its output artifacts.

    The default call performs a real UHD capture.  Supplying ``usrp`` and an
    alternate ``capture_function`` permits loopback/replay testing while using
    the identical frame, receiver, aggregation, save, and plot pipeline.
    """
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if usrp is None:
        usrp = _initialize_usrp(config)

    experiment_outputs: list[ExperimentArtifacts] = []
    for bandwidth_config in iter_capture_configs(config):
        print("=" * 80)
        print(
            f"Running bandwidth setting: "
            f"{bandwidth_config.BANDWIDTH} MHz"
        )
        print(
            f"Virtual pilot count: "
            f"{bandwidth_config.NUM_VIRTUAL_PILOTS} | "
            "expected channel-estimation gain: "
            f"{10 * np.log10(bandwidth_config.NUM_VIRTUAL_PILOTS):.2f} dB"
        )

        known_ref_seq, frame_artifacts = build_frame_from_config(
            bandwidth_config
        )
        results = collect_channel_results(
            usrp,
            frame_artifacts,
            known_ref_seq,
            bandwidth_config,
            capture_function=capture_function,
        )
        npz_path, json_path, figure_path, metadata = save_results(
            bandwidth_config.OUTPUT_DIR,
            bandwidth_config,
            known_ref_seq,
            results,
        )
        plot_paths = plot_results(
            figure_path,
            bandwidth_config,
            results,
            show=show_plots,
        )
        _print_saved_summary(
            npz_path,
            json_path,
            plot_paths,
            metadata,
        )
        experiment_outputs.append(
            {
                "config": bandwidth_config,
                "results": results,
                "npz_path": npz_path,
                "json_path": json_path,
                "plot_paths": plot_paths,
                "metadata": metadata,
            }
        )

    return experiment_outputs


def main() -> None:
    run_capture_experiment()


if __name__ == "__main__":
    main()
