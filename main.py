from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from channel import apply_channel, frequency_response_at_bins, make_example_channels
from config import SoundingConfig
from receiver import receive_frame
from transmitter import build_transmit_frame


def nmse_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    error_power = np.sum(np.abs(estimate - reference) ** 2)
    reference_power = np.sum(np.abs(reference) ** 2)
    return float(10.0 * np.log10(error_power / reference_power))


def main() -> None:
    cfg = SoundingConfig()
    cfg.validate()

    tx = build_transmit_frame(cfg)
    taps = make_example_channels(cfg)
    channel_output = apply_channel(tx.branch_samples, taps, cfg)
    rx = receive_frame(
        channel_output.received,
        tx.zc,
        tx.pilot_symbols,
        cfg,
    )

    true_channel = frequency_response_at_bins(taps, cfg.pilot_bins, cfg.nfft)
    individual_error_power = np.mean(
        np.sum(
            np.abs(
                rx.per_pilot_channel_estimates - true_channel[np.newaxis, :]
            ) ** 2,
            axis=1,
        )
    )
    reference_power = np.sum(np.abs(true_channel) ** 2)
    mean_individual_nmse = float(
        10.0 * np.log10(individual_error_power / reference_power)
    )
    averaged_nmse = nmse_db(rx.channel_estimates, true_channel)

    print("=== Paper-aligned waveform ===")
    print(f"Center frequency : {cfg.center_frequency_hz / 1e9:.3f} GHz")
    print(f"Nominal bandwidth: {cfg.nominal_bandwidth_hz / 1e6:.3f} MHz")
    print(f"Subcarrier spacing: {cfg.subcarrier_spacing_hz / 1e3:.3f} kHz")
    print(f"FFT / CP samples : {cfg.nfft} / {cfg.cp_len}")
    print(f"Sample rate      : {cfg.sample_rate_hz / 1e6:.3f} MS/s")
    print(f"Pilot period     : {cfg.pilot_period_s * 1e3:.3f} ms")
    print(f"Pilots/estimate  : {cfg.pilot_repetitions}")
    print(f"Period samples   : {cfg.period_samples}")
    print(f"Burst samples    : {cfg.burst_samples}")
    print()
    print("=== Synchronization ===")
    print(f"True timing      : {cfg.timing_offset_samples} samples")
    print(f"Estimated timing : {rx.sync_start} samples")
    print(f"True CFO         : {cfg.cfo_hz:.3f} Hz")
    print(f"Estimated CFO    : {rx.cfo_estimate_hz:.3f} Hz")
    print()
    print("=== Sparse channel estimates ===")
    print("branch  bin   frequency(GHz)       true H              estimated H")
    for branch, (bin_index, h_true, h_est) in enumerate(
        zip(cfg.pilot_bins, true_channel, rx.channel_estimates)
    ):
        frequency = (
            cfg.center_frequency_hz + bin_index * cfg.subcarrier_spacing_hz
        ) / 1e9
        print(
            f"{branch:>3d}  {bin_index:>4d}    {frequency:.7f}    "
            f"{h_true.real:+.4f}{h_true.imag:+.4f}j    "
            f"{h_est.real:+.4f}{h_est.imag:+.4f}j"
        )
    print(f"Mean per-pilot NMSE: {mean_individual_nmse:.2f} dB")
    print(f"3-pilot avg. NMSE  : {averaged_nmse:.2f} dB")

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    left = max(0, rx.sync_start - 80)
    right = min(rx.sync_metric.size, rx.sync_start + 160)
    axes[0].plot(np.arange(left, right), rx.sync_metric[left:right])
    axes[0].axvline(rx.sync_start, color="tab:red", linestyle="--")
    axes[0].set_title("Normalized ZC correlation")
    axes[0].set_xlabel("Candidate frame-start sample")
    axes[0].set_ylabel("Metric")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        true_channel.real,
        true_channel.imag,
        marker="o",
        s=65,
        label="True channel",
    )
    for repetition, estimate in enumerate(rx.per_pilot_channel_estimates):
        axes[1].scatter(
            estimate.real,
            estimate.imag,
            marker=".",
            color="0.55",
            alpha=0.55,
            label="Individual pilots" if repetition == 0 else None,
        )
    axes[1].scatter(
        rx.channel_estimates.real,
        rx.channel_estimates.imag,
        marker="x",
        s=70,
        label="3-pilot LS average",
    )
    for branch, h in enumerate(rx.channel_estimates):
        axes[1].annotate(
            str(branch),
            (h.real, h.imag),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[1].set_title("Sparse complex channel coefficients")
    axes[1].set_xlabel("Real")
    axes[1].set_ylabel("Imaginary")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    figure_path = output_dir / "sounding_demo.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
