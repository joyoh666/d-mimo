#!/usr/bin/env python3
"""Stable fixed-batch GPU benchmark for the fixed-SCS wideband GRU models.

The training experiment's test set can end in a small final batch, which makes
GPU utilization dominate latency.  This script repeats held-out sequences to a
common batch size (128 by default), performs warm-up iterations, and reports the
median of five timed trials for every bandwidth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fixed_scs_wideband_gru_experiment import (
    WidebandChannelGRU,
    WidebandConfig,
    bandwidth_tag,
    checkpoint_path,
    load_data_and_indices,
    make_loaders,
    rollout,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark(
    model: WidebandChannelGRU,
    sequence: torch.Tensor,
    cfg: WidebandConfig,
    device: torch.device,
    warmups: int,
    repeats: int,
    trials: int,
) -> dict[str, float]:
    model.eval()
    t0 = (sequence.shape[1] - cfg.pred_steps) // 2
    for _ in range(warmups):
        rollout(model, sequence, t0, cfg.pred_steps)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(repeats):
            rollout(model, sequence, t0, cfg.pred_steps)
        synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0 / repeats)

    latency_ms = float(np.median(latencies_ms))
    peak_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    return {
        "latency_ms_per_batch": latency_ms,
        "throughput_sequences_s": sequence.shape[0] * 1000.0 / latency_ms,
        "peak_memory_mb": peak_bytes / (1024.0**2),
        "latency_trial_min_ms": float(np.min(latencies_ms)),
        "latency_trial_max_ms": float(np.max(latencies_ms)),
    }


def read_training_times(summary_path: Path) -> dict[float, float]:
    result = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = row.get("training_seconds", "")
                if value not in ("", "None", None):
                    result[float(row["bandwidth_mhz"])] = float(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/fixed_scs_wideband_gru")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    with (output_dir / "config.json").open("r", encoding="utf-8") as handle:
        cfg = WidebandConfig(**json.load(handle))
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    training_times = read_training_times(output_dir / "summary.csv")

    rows = []
    for bandwidth_mhz in cfg.bandwidths_mhz:
        array, indices = load_data_and_indices(cfg, output_dir, bandwidth_mhz)
        loader = make_loaders(array, indices, cfg, device)["test"]
        base = next(iter(loader))
        copies = math.ceil(args.batch_size / base.shape[0])
        sequence = base.repeat((copies, 1, 1))[: args.batch_size].to(device)

        checkpoint = torch.load(
            checkpoint_path(output_dir, bandwidth_mhz),
            map_location="cpu",
            weights_only=False,
        )
        model = WidebandChannelGRU(int(checkpoint["token_dim"]), cfg).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics = benchmark(
            model,
            sequence,
            cfg,
            device,
            args.warmups,
            args.repeats,
            args.trials,
        )
        row = {
            "bandwidth_mhz": bandwidth_mhz,
            "num_subcarriers": cfg.num_subcarriers(bandwidth_mhz),
            "token_dim": model.token_dim,
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": training_times.get(bandwidth_mhz),
            "benchmark_batch_size": args.batch_size,
            **metrics,
        }
        rows.append(row)
        print(
            f"[BENCH] {bandwidth_mhz:5g} MHz Nsc={row['num_subcarriers']:4d} "
            f"latency={metrics['latency_ms_per_batch']:8.3f} ms "
            f"throughput={metrics['throughput_sequences_s']:9.1f} seq/s"
        )

    reference = rows[0]
    for row in rows:
        row["latency_ratio_vs_reference"] = (
            row["latency_ms_per_batch"] / reference["latency_ms_per_batch"]
        )
        row["parameter_ratio_vs_reference"] = (
            row["num_parameters"] / reference["num_parameters"]
        )
        if row["training_seconds"] is not None and reference["training_seconds"]:
            row["training_time_ratio_vs_reference"] = (
                row["training_seconds"] / reference["training_seconds"]
            )
        else:
            row["training_time_ratio_vs_reference"] = None

    fields = list(rows[0].keys())
    csv_path = output_dir / f"inference_benchmark_batch{args.batch_size}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    bandwidths = [row["bandwidth_mhz"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(
        bandwidths,
        [row["latency_ms_per_batch"] for row in rows],
        "o-",
        linewidth=2,
    )
    axes[0].set_title(f"5-step inference, batch={args.batch_size}")
    axes[0].set_ylabel("Latency (ms / batch)")
    axes[1].plot(
        bandwidths,
        [row["training_seconds"] for row in rows],
        "s-",
        linewidth=2,
        color="tab:orange",
    )
    axes[1].set_title("Training wall time")
    axes[1].set_ylabel("Seconds")
    axes[2].plot(
        bandwidths,
        [row["num_parameters"] / 1e6 for row in rows],
        "^-",
        linewidth=2,
        color="tab:green",
    )
    axes[2].set_title("Model size")
    axes[2].set_ylabel("Parameters (million)")
    for axis in axes:
        axis.set_xlabel("Bandwidth (MHz), SCS=15 kHz")
        axis.set_xticks(bandwidths)
        axis.grid(True, alpha=0.3)
    figure.suptitle("Fixed-SCS wideband GRU compute scaling")
    figure.tight_layout()
    plot_path = output_dir / "fixed_scs_compute_benchmark.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f"[CSV] {csv_path}")
    print(f"[PLOT] {plot_path}")


if __name__ == "__main__":
    main()
