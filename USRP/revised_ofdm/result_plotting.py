"""Plot generation for OFDM channel-capture results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:  # Support package-style imports.
    from .capture_pipeline import CaptureResults
    from .config import DEFAULT_CONFIG, OFDMConfig
    from ...modulate import modulations
    from .receiver_processing import delay_residual_power
else:  # Support direct execution from the USRP directory.
    from USRP.test.revised_ofdm.capture_pipeline import CaptureResults
    from USRP.test.revised_ofdm.config import DEFAULT_CONFIG, OFDMConfig
    from modulate import modulations
    from USRP.test.revised_ofdm.receiver_processing import delay_residual_power


PlotPaths = dict[str, str]


def _related_figure_path(figure_path: str | Path, prefix: str) -> Path:
    path = Path(figure_path)
    return path.with_name(path.name.replace("channel_plots_", prefix, 1))


def _display_and_close(figure: plt.Figure, *, show: bool) -> None:
    if show:
        plt.show()
    plt.close(figure)


def plot_results(
    figure_path: str | Path,
    config: OFDMConfig,
    results: CaptureResults,
    *,
    show: bool = True,
) -> PlotPaths:
    """Create the same summary, delay, constellation, and PSD plots.

    As in the original program, the constellation figure is displayed but is
    not written to disk.  ``show=False`` is useful for unattended/headless
    runs while retaining all three saved PNG files.
    """
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    delay_figure_path = _related_figure_path(
        figure_path,
        "channel_plots_delay_",
    )
    equalized_psd_path = _related_figure_path(
        figure_path,
        "channel_plots_eqpsd_",
    )

    channel_mean_fd = results["channel_mean_fd"]
    impulse_response = results["channel_impulse_response_td"]
    single_impulse_response = (
        results["channel_impulse_response_td_single_pilot"]
    )
    example_iq_rcv = results["example_iq_rcv"]
    example_iq_rcv_single = results["example_iq_rcv_single"]
    example_frame_rcv = results["example_frame_rcv"]
    virtual_pilot_snr_gain_db = results["virtual_pilot_snr_gain_db"]
    variance_raw = results["virtual_pilot_variance_raw"]
    variance_aligned = results["virtual_pilot_variance_aligned"]
    delay_residual_gain_db = results["delay_profile_residual_gain_db"]

    average_abs_channel = np.mean(np.abs(channel_mean_fd), axis=0)
    average_abs_delay = np.mean(np.abs(impulse_response), axis=0)
    average_abs_delay_single = np.mean(
        np.abs(single_impulse_response),
        axis=0,
    )
    average_variance_raw = np.mean(variance_raw, axis=0)
    average_variance_aligned = np.mean(variance_aligned, axis=0)
    average_gain_db = 10 * np.log10(
        (average_variance_raw + 1e-15)
        / (
            average_variance_aligned
            / max(config.NUM_VIRTUAL_PILOTS, 1)
            + 1e-15
        )
    )
    measured_gain_db = float(np.mean(virtual_pilot_snr_gain_db))

    delay_plot_guard = max(1, int(np.ceil(config.FFT_SIZE / 500)))
    single_main_power, single_residual_power, _ = delay_residual_power(
        average_abs_delay_single,
        guard_taps=delay_plot_guard,
    )
    averaged_main_power, averaged_residual_power, _ = delay_residual_power(
        average_abs_delay,
        guard_taps=delay_plot_guard,
    )
    measured_delay_gain_db = float(np.mean(delay_residual_gain_db))

    figure, axes = plt.subplots(2, 3, figsize=(18, 9))
    figure.suptitle(
        f"Channel capture summary | BW={config.BANDWIDTH} MHz | "
        f"N={config.N} | FFT={config.FFT_SIZE} | "
        f"captures={channel_mean_fd.shape[0]} | "
        f"VP={config.NUM_VIRTUAL_PILOTS} | "
        f"measured pilot gain={measured_gain_db:.2f} dB"
    )

    axes[0, 0].plot(average_abs_channel)
    axes[0, 0].set_title("Average |H[k]| across captures")
    axes[0, 0].set_xlabel("Subcarrier index")
    axes[0, 0].set_ylabel("Magnitude")
    axes[0, 0].grid(True, alpha=0.3)

    channel_image = axes[0, 1].imshow(
        np.abs(channel_mean_fd),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    axes[0, 1].set_title("|H[k]| for repeated captures")
    axes[0, 1].set_xlabel("Subcarrier index")
    axes[0, 1].set_ylabel("Capture index")
    figure.colorbar(channel_image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].plot(average_abs_delay)
    axes[0, 2].set_title("Average |h[n]| in delay domain (IFFT of channel)")
    axes[0, 2].set_xlabel("Delay sample")
    axes[0, 2].set_ylabel("Magnitude")
    axes[0, 2].grid(True, alpha=0.3)

    delay_figure, delay_axes = plt.subplots(1, 2, figsize=(12, 5))
    delay_axes[0].plot(
        average_abs_delay_single,
        label="Single-pilot delay |h[n]|",
    )
    delay_axes[0].plot(
        average_abs_delay,
        linestyle="--",
        label=(
            f"{config.NUM_VIRTUAL_PILOTS}-pilot averaged delay |h[n]|"
        ),
    )
    delay_axes[0].set_title("Delay-domain mean profile")
    delay_axes[0].set_xlabel("Delay sample")
    delay_axes[0].set_ylabel("Magnitude")
    delay_axes[0].grid(True, alpha=0.3)
    delay_axes[0].legend()

    delay_axes[1].text(
        0.05,
        0.82,
        "Single residual power: "
        f"{10 * np.log10(single_residual_power + 1e-15):.2f} dB",
        transform=delay_axes[1].transAxes,
    )
    delay_axes[1].text(
        0.05,
        0.65,
        "Averaged residual power: "
        f"{10 * np.log10(averaged_residual_power + 1e-15):.2f} dB",
        transform=delay_axes[1].transAxes,
    )
    delay_axes[1].text(
        0.05,
        0.48,
        "Residual suppression gain: "
        f"{10 * np.log10(single_residual_power + 1e-15) - 10 * np.log10(averaged_residual_power + 1e-15):.2f} dB",
        transform=delay_axes[1].transAxes,
    )
    delay_axes[1].text(
        0.05,
        0.31,
        "Delay mean |h| peak: "
        f"single={single_main_power:.3e}, "
        f"averaged={averaged_main_power:.3e}",
        transform=delay_axes[1].transAxes,
    )
    delay_axes[1].axis("off")
    delay_axes[1].set_title("Delay-domain residual metric")
    delay_figure.suptitle(
        "Delay profile comparison | measured delay residual suppression="
        f"{measured_delay_gain_db:.2f} dB"
    )
    delay_figure.tight_layout()
    delay_figure.savefig(delay_figure_path, dpi=150, bbox_inches="tight")
    _display_and_close(delay_figure, show=show)

    axes[1, 0].plot(
        10 * np.log10(average_variance_raw + 1e-15),
        label="Raw virtual-pilot variance",
    )
    axes[1, 0].plot(
        10 * np.log10(average_variance_aligned + 1e-15),
        label="Phase-aligned virtual-pilot variance",
    )
    axes[1, 0].set_title("Virtual pilot estimator variance")
    axes[1, 0].set_xlabel("Subcarrier index")
    axes[1, 0].set_ylabel("Variance (dB)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(
        10 * np.log10(average_variance_raw + 1e-15),
        label="Raw pilot variance",
    )
    axes[1, 1].plot(
        10 * np.log10(average_variance_aligned + 1e-15),
        label="Aligned pilot variance",
    )
    # Keep the transformation used by the original figure for compatibility.
    axes[1, 1].plot(
        10 * np.log10(average_gain_db + 1e-15),
        label="Estimated gain from averaging",
    )
    axes[1, 1].set_title("Variance reduction from averaging")
    axes[1, 1].set_xlabel("Subcarrier index")
    axes[1, 1].set_ylabel("dB")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].psd(
        example_frame_rcv,
        NFFT=example_frame_rcv.size,
        Fs=config.sampling_rate,
        scale_by_freq=False,
    )
    axes[1, 2].set_title("Example received-frame spectrum")
    axes[1, 2].set_xlabel("Frequency (Hz)")
    axes[1, 2].set_ylabel("Power spectrum (dB)")

    figure.tight_layout()
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    _display_and_close(figure, show=show)

    constellation_figure, constellation_axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )
    real_constellation = np.real(modulations[config.modulation_order])
    peak_amplitude = 2 * np.max(np.abs(real_constellation))
    constellation_axes[0].scatter(
        np.real(example_iq_rcv_single),
        np.imag(example_iq_rcv_single),
        s=1.0,
        marker="o",
    )
    constellation_axes[0].set_xlim([-peak_amplitude, peak_amplitude])
    constellation_axes[0].set_ylim([-peak_amplitude, peak_amplitude])
    constellation_axes[0].set_xlabel("In-Phase")
    constellation_axes[0].set_ylabel("Quadrature")
    constellation_axes[0].set_title("Single-pilot equalized constellation")
    constellation_axes[0].grid(True, alpha=0.3)

    constellation_axes[1].scatter(
        np.real(example_iq_rcv),
        np.imag(example_iq_rcv),
        s=1.0,
        marker="o",
    )
    constellation_axes[1].set_xlim([-peak_amplitude, peak_amplitude])
    constellation_axes[1].set_ylim([-peak_amplitude, peak_amplitude])
    constellation_axes[1].set_xlabel("In-Phase")
    constellation_axes[1].set_ylabel("Quadrature")
    constellation_axes[1].set_title(
        f"{config.NUM_VIRTUAL_PILOTS}-pilot averaged equalized constellation"
    )
    constellation_axes[1].grid(True, alpha=0.3)

    constellation_figure.tight_layout()
    _display_and_close(constellation_figure, show=show)

    equalized_psd_figure, psd_axes = plt.subplots(1, 2, figsize=(12, 5))
    single_nfft = min(
        4096,
        max(
            256,
            1 << int(np.ceil(np.log2(max(1, example_iq_rcv_single.size)))),
        ),
    )
    averaged_nfft = min(
        4096,
        max(
            256,
            1 << int(np.ceil(np.log2(max(1, example_iq_rcv.size)))),
        ),
    )
    psd_axes[0].psd(
        example_iq_rcv_single,
        NFFT=single_nfft,
        Fs=config.sampling_rate,
        scale_by_freq=False,
    )
    psd_axes[0].set_title("Single-pilot equalized PSD")
    psd_axes[0].set_xlabel("Frequency (Hz)")
    psd_axes[0].set_ylabel("Power spectrum (dB)")
    psd_axes[0].grid(True, alpha=0.3)

    psd_axes[1].psd(
        example_iq_rcv,
        NFFT=averaged_nfft,
        Fs=config.sampling_rate,
        scale_by_freq=False,
    )
    psd_axes[1].set_title(
        f"{config.NUM_VIRTUAL_PILOTS}-pilot averaged equalized PSD"
    )
    psd_axes[1].set_xlabel("Frequency (Hz)")
    psd_axes[1].set_ylabel("Power spectrum (dB)")
    psd_axes[1].grid(True, alpha=0.3)

    equalized_psd_figure.suptitle(
        "Equalized spectrum | measured equalization gain="
        f"{measured_gain_db:.2f} dB | VP={config.NUM_VIRTUAL_PILOTS}"
    )
    equalized_psd_figure.tight_layout()
    equalized_psd_figure.savefig(
        equalized_psd_path,
        dpi=150,
        bbox_inches="tight",
    )
    _display_and_close(equalized_psd_figure, show=show)

    return {
        "channel_plots": str(figure_path),
        "delay_profile": str(delay_figure_path),
        "equalized_psd": str(equalized_psd_path),
    }


def plot_default_results(
    figure_path: str | Path,
    results: CaptureResults,
    config: OFDMConfig = DEFAULT_CONFIG,
    *,
    show: bool = True,
) -> PlotPaths:
    """Plot using the default configuration."""
    return plot_results(figure_path, config, results, show=show)
