#!/usr/bin/env python3
"""Controlled bandwidth sweep for the existing five-step GRU predictor.

This is a standalone experiment.  It does not import from or modify the
existing training/test notebooks.  The model, tokenization, and rollout match
``Final_Train_260217_RNN_fivestep.ipynb`` as closely as practical:

* one complex CSI sample every 5 ms;
* five samples per token (25 ms);
* gain/cos/sin tokens with gain scale 20;
* one-layer GRU with 64 hidden units and an age feature;
* five autoregressive prediction steps with gamma=0.5 loss weights.

The simulated channel is 3GPP TDL-A at 2.2 GHz.  A single set of underlying
path coefficients, path delays, data splits, and standardized noise samples is
shared by all bandwidths.  Only the six pilot frequency offsets change.  This
makes the sweep a controlled test of bandwidth rather than four unrelated
Monte-Carlo runs.

Examples
--------
Full default experiment (generate, train, test, and plot)::

    .venv/bin/python bandwidth_gru_experiment.py --mode all

Quick smoke run::

    .venv/bin/python bandwidth_gru_experiment.py --mode all \
        --windows 48 --epochs 2 --output-dir output/bandwidth_gru_smoke

Re-test saved best checkpoints without retraining::

    .venv/bin/python bandwidth_gru_experiment.py --mode test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_BANDWIDTHS_MHZ = (1.4, 5.0, 9.0, 15.0)
PILOT_FRACTIONS = (-0.45, -0.27, -0.09, 0.09, 0.27, 0.45)


@dataclass(frozen=True)
class ExperimentConfig:
    bandwidths_mhz: tuple[float, ...]
    carrier_frequency_hz: float
    delay_spread_s: float
    speed_mps: float
    snr_db: float
    windows: int
    samples_per_window: int
    samples_per_step: int
    csi_period_s: float
    tdl_batch_size: int
    train_fraction: float
    val_fraction: float
    seed: int
    hidden_size: int
    num_layers: int
    dropout: float
    gain_scale: float
    eps_norm: float
    pred_steps: int
    loss_gamma: float
    max_age_feature: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    patience: int
    min_epochs: int

    @property
    def token_dim(self) -> int:
        return 3 * self.samples_per_step

    @property
    def steps_per_window(self) -> int:
        return self.samples_per_window // self.samples_per_step


def parse_bandwidths(text: str) -> tuple[float, ...]:
    values = tuple(float(v.strip()) for v in text.split(",") if v.strip())
    if not values or any((not np.isfinite(v)) or v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Bandwidths must be positive numbers in MHz.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Bandwidth values must be unique.")
    return values


def bandwidth_tag(bandwidth_mhz: float) -> str:
    return f"{bandwidth_mhz:g}mhz".replace(".", "p")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is unavailable.")
    return device


def split_indices(cfg: ExperimentConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(cfg.windows)
    n_train = int(math.floor(cfg.train_fraction * cfg.windows))
    n_val = int(math.floor(cfg.val_fraction * cfg.windows))
    n_test = cfg.windows - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            f"Split is empty: train={n_train}, val={n_val}, test={n_test}. "
            "Increase --windows."
        )
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def _validate_config(cfg: ExperimentConfig) -> None:
    if cfg.samples_per_window % cfg.samples_per_step != 0:
        raise ValueError("samples-per-window must be divisible by samples-per-step.")
    if cfg.steps_per_window < cfg.pred_steps + 2:
        raise ValueError("Each window needs at least pred-steps + 2 tokens.")
    if cfg.samples_per_step < 1 or cfg.pred_steps < 1:
        raise ValueError("samples-per-step and pred-steps must be positive.")
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    if cfg.delay_spread_s <= 0 or cfg.csi_period_s <= 0:
        raise ValueError("delay-spread and CSI period must be positive.")
    if cfg.speed_mps < 0:
        raise ValueError("speed must be non-negative.")


def generate_datasets(cfg: ExperimentConfig, output_dir: Path, device: torch.device) -> None:
    """Generate matched TDL-A observations for every requested bandwidth."""
    try:
        from sionna.phy.channel.tr38901 import TDL
    except ImportError as exc:
        raise RuntimeError(
            "Sionna is required for generation. Run with this project's .venv."
        ) from exc

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        bw: np.empty(
            (cfg.windows, cfg.samples_per_window, len(PILOT_FRACTIONS)),
            dtype=np.complex64,
        )
        for bw in cfg.bandwidths_mhz
    }

    set_all_seeds(cfg.seed)
    tdl = TDL(
        model="A",
        delay_spread=cfg.delay_spread_s,
        carrier_frequency=cfg.carrier_frequency_hz,
        min_speed=cfg.speed_mps,
        max_speed=cfg.speed_mps,
        num_rx_ant=1,
        num_tx_ant=len(PILOT_FRACTIONS),
        device=str(device),
    )
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(cfg.seed + 17_029)

    print(
        f"[GENERATE] TDL-A, fc={cfg.carrier_frequency_hz/1e9:.3f} GHz, "
        f"DS={cfg.delay_spread_s*1e9:.1f} ns, speed={cfg.speed_mps:.2f} m/s, "
        f"SNR={cfg.snr_db:.1f} dB"
    )
    start = 0
    while start < cfg.windows:
        batch = min(cfg.tdl_batch_size, cfg.windows - start)
        h, delays = tdl(
            batch_size=batch,
            num_time_steps=cfg.samples_per_window,
            sampling_frequency=1.0 / cfg.csi_period_s,
        )
        # h: [B,1,1,1,6,path,time], delays: [B,1,1,path]
        h = h[:, 0, 0, 0, :, :, :]
        tau = delays[:, 0, 0, :]
        if h.ndim != 4 or h.shape[1] != len(PILOT_FRACTIONS):
            raise RuntimeError(f"Unexpected TDL coefficient shape: {tuple(h.shape)}")
        if tau.ndim != 2 or tau.shape[1] != h.shape[2]:
            raise RuntimeError(
                f"Unexpected TDL delay shape {tuple(tau.shape)} for h {tuple(h.shape)}"
            )

        noise_standard = torch.complex(
            torch.randn(
                (batch, len(PILOT_FRACTIONS), cfg.samples_per_window),
                generator=noise_generator,
                device=device,
            ),
            torch.randn(
                (batch, len(PILOT_FRACTIONS), cfg.samples_per_window),
                generator=noise_generator,
                device=device,
            ),
        ) / math.sqrt(2.0)

        for bw_mhz in cfg.bandwidths_mhz:
            offsets_hz = torch.tensor(
                [fraction * bw_mhz * 1e6 for fraction in PILOT_FRACTIONS],
                dtype=tau.dtype,
                device=device,
            )
            phase = torch.exp(
                -1j
                * 2.0
                * math.pi
                * offsets_hz.view(1, -1, 1)
                * tau.unsqueeze(1)
            )
            cfr = 0.05 * (h * phase.unsqueeze(-1)).sum(dim=2)

            # Keep the observation SNR identical for every bandwidth.
            signal_power = cfr.abs().square().mean(dim=-1, keepdim=True)
            noise_power = signal_power / (10.0 ** (cfg.snr_db / 10.0))
            observed = cfr + noise_standard * noise_power.clamp_min(1e-12).sqrt()
            arrays[bw_mhz][start : start + batch] = (
                observed.permute(0, 2, 1).detach().cpu().numpy().astype(np.complex64)
            )

        start += batch
        print(f"[GENERATE] {start:5d}/{cfg.windows} matched windows", flush=True)

    splits = split_indices(cfg)
    for bw_mhz, values in arrays.items():
        path = data_dir / f"channel_{bandwidth_tag(bw_mhz)}.npz"
        np.savez_compressed(
            path,
            X=values,
            bandwidth_mhz=np.float32(bw_mhz),
            pilot_fractions=np.asarray(PILOT_FRACTIONS, dtype=np.float32),
            pilot_offsets_hz=np.asarray(PILOT_FRACTIONS, dtype=np.float64)
            * bw_mhz
            * 1e6,
            dt=np.float32(cfg.csi_period_s),
            carrier_frequency_hz=np.float64(cfg.carrier_frequency_hz),
            delay_spread_s=np.float64(cfg.delay_spread_s),
            speed_mps=np.float32(cfg.speed_mps),
            snr_db=np.float32(cfg.snr_db),
            train_indices=splits["train"],
            val_indices=splits["val"],
            test_indices=splits["test"],
        )
        print(f"[DATA] saved {path} X={values.shape}")


def complex_to_tokens(h: torch.Tensor, gain_scale: float) -> torch.Tensor:
    magnitude = h.abs() * gain_scale
    angle = torch.angle(h)
    return torch.cat((magnitude, torch.cos(angle), torch.sin(angle)), dim=-1).float()


def postprocess_tokens(raw: torch.Tensor, taps: int, eps_norm: float) -> torch.Tensor:
    gain = F.softplus(raw[..., :taps])
    cos_raw = raw[..., taps : 2 * taps]
    sin_raw = raw[..., 2 * taps : 3 * taps]
    norm = torch.sqrt(cos_raw.square() + sin_raw.square() + eps_norm)
    return torch.cat((gain, cos_raw / norm, sin_raw / norm), dim=-1)


def tokens_to_complex(tokens: torch.Tensor, taps: int, gain_scale: float) -> torch.Tensor:
    gain = tokens[..., :taps] / gain_scale
    cos = tokens[..., taps : 2 * taps]
    sin = tokens[..., 2 * taps : 3 * taps]
    return torch.complex(gain * cos, gain * sin)


class PerLinkTokenDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        x: np.ndarray,
        window_indices: np.ndarray,
        samples_per_step: int,
        gain_scale: float,
        augment_phase: bool,
    ) -> None:
        if x.ndim != 3 or x.shape[-1] != len(PILOT_FRACTIONS):
            raise ValueError(f"Expected X [window,time,6], got {x.shape}")
        self.x = x
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.samples_per_step = samples_per_step
        self.gain_scale = gain_scale
        self.augment_phase = augment_phase
        self.steps = x.shape[1] // samples_per_step

    def __len__(self) -> int:
        return len(self.window_indices) * len(PILOT_FRACTIONS)

    def __getitem__(self, index: int) -> torch.Tensor:
        local_window, link = divmod(index, len(PILOT_FRACTIONS))
        window = int(self.window_indices[local_window])
        h = torch.from_numpy(np.array(self.x[window, :, link], copy=True))
        if self.augment_phase:
            phase = torch.rand((), dtype=torch.float32) * (2.0 * math.pi)
            h = h * torch.polar(torch.ones((), dtype=torch.float32), phase)
        h = h.reshape(self.steps, self.samples_per_step)
        return complex_to_tokens(h, self.gain_scale)


class SimpleChannelGRU(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.token_dim = cfg.token_dim
        self.max_age_feature = cfg.max_age_feature
        self.eps_norm = cfg.eps_norm
        self.taps = cfg.samples_per_step
        self.init_token = nn.Parameter(torch.zeros(cfg.token_dim))
        nn.init.normal_(self.init_token, std=0.02)
        self.gru = nn.GRU(
            input_size=cfg.token_dim + 1,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(cfg.hidden_size, cfg.token_dim)

    def step(
        self, token: torch.Tensor, age: int, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        age_value = min(age, self.max_age_feature) / float(self.max_age_feature)
        age_feature = torch.full(
            (token.shape[0], 1), age_value, device=token.device, dtype=token.dtype
        )
        output, hidden = self.gru(torch.cat((token, age_feature), dim=-1).unsqueeze(1), hidden)
        pred = postprocess_tokens(self.output(output[:, 0]), self.taps, self.eps_norm)
        return pred, hidden


def warmup_hidden(model: SimpleChannelGRU, seq: torch.Tensor, t0: int) -> torch.Tensor:
    batch, _, dim = seq.shape
    init = model.init_token.view(1, 1, dim).expand(batch, 1, dim)
    if t0 > 0:
        prefix = torch.cat((init, seq[:, :t0]), dim=1)
    else:
        prefix = init
    age_zeros = torch.zeros((batch, prefix.shape[1], 1), device=seq.device, dtype=seq.dtype)
    _, hidden = model.gru(torch.cat((prefix, age_zeros), dim=-1))
    return hidden


def rollout(
    model: SimpleChannelGRU, seq: torch.Tensor, t0: int, pred_steps: int
) -> torch.Tensor:
    hidden = warmup_hidden(model, seq, t0)
    current = seq[:, t0]
    predictions = []
    for step in range(pred_steps):
        pred, hidden = model.step(current, step, hidden)
        predictions.append(pred)
        current = pred.detach()
    return torch.stack(predictions, dim=1)


def training_loss(
    model: SimpleChannelGRU, seq: torch.Tensor, cfg: ExperimentConfig
) -> torch.Tensor:
    last_t0 = seq.shape[1] - cfg.pred_steps - 1
    t0 = int(torch.randint(0, last_t0 + 1, (1,), device=seq.device).item())
    pred = rollout(model, seq, t0, cfg.pred_steps)
    target = seq[:, t0 + 1 : t0 + 1 + cfg.pred_steps]
    weights = torch.tensor(
        [cfg.loss_gamma**i for i in range(cfg.pred_steps)],
        device=seq.device,
        dtype=seq.dtype,
    )
    per_step = (pred - target).square().mean(dim=(0, 2))
    return (per_step * weights).sum() / weights.sum()


@torch.no_grad()
def evaluate(
    model: SimpleChannelGRU,
    loader: DataLoader[torch.Tensor],
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    error_by_step = torch.zeros(cfg.pred_steps, dtype=torch.float64)
    energy_by_step = torch.zeros(cfg.pred_steps, dtype=torch.float64)
    persistence_error_by_step = torch.zeros(cfg.pred_steps, dtype=torch.float64)
    token_square_error = 0.0
    token_count = 0
    sequences = 0
    t0_count = 0

    for seq in loader:
        seq = seq.to(device, non_blocking=device.type == "cuda")
        batch = seq.shape[0]
        for t0 in range(seq.shape[1] - cfg.pred_steps):
            pred_token = rollout(model, seq, t0, cfg.pred_steps)
            target_token = seq[:, t0 + 1 : t0 + 1 + cfg.pred_steps]
            pred = tokens_to_complex(pred_token, cfg.samples_per_step, cfg.gain_scale)
            target = tokens_to_complex(target_token, cfg.samples_per_step, cfg.gain_scale)
            context = tokens_to_complex(
                seq[:, t0], cfg.samples_per_step, cfg.gain_scale
            ).unsqueeze(1)
            persistence = context.expand(-1, cfg.pred_steps, -1)

            error_by_step += (pred - target).abs().square().sum(dim=(0, 2)).cpu()
            energy_by_step += target.abs().square().sum(dim=(0, 2)).cpu()
            persistence_error_by_step += (
                (persistence - target).abs().square().sum(dim=(0, 2)).cpu()
            )
            token_square_error += float((pred_token - target_token).square().sum().item())
            token_count += pred_token.numel()
            sequences += batch
            t0_count += 1

    if sequences == 0:
        raise RuntimeError("Evaluation loader produced no sequences.")
    weights = torch.tensor(
        [cfg.loss_gamma**i for i in range(cfg.pred_steps)], dtype=torch.float64
    )
    weighted_error = (error_by_step * weights).sum()
    weighted_energy = (energy_by_step * weights).sum().clamp_min(1e-15)
    persistence_weighted_error = (persistence_error_by_step * weights).sum()
    nmse = float((weighted_error / weighted_energy).item())
    persistence_nmse = float((persistence_weighted_error / weighted_energy).item())
    per_step_nmse = (error_by_step / energy_by_step.clamp_min(1e-15)).numpy()
    persistence_per_step_nmse = (
        persistence_error_by_step / energy_by_step.clamp_min(1e-15)
    ).numpy()
    return {
        "token_mse": token_square_error / token_count,
        "nmse": nmse,
        "nmse_db": 10.0 * math.log10(max(nmse, 1e-15)),
        "per_step_nmse": per_step_nmse.tolist(),
        "per_step_nmse_db": (10.0 * np.log10(np.maximum(per_step_nmse, 1e-15))).tolist(),
        "persistence_nmse": persistence_nmse,
        "persistence_nmse_db": 10.0 * math.log10(max(persistence_nmse, 1e-15)),
        "persistence_per_step_nmse": persistence_per_step_nmse.tolist(),
        "sequences": sequences,
        "t0_per_sequence": t0_count,
    }


def load_bandwidth_data(
    cfg: ExperimentConfig, output_dir: Path, bandwidth_mhz: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = output_dir / "data" / f"channel_{bandwidth_tag(bandwidth_mhz)}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run with --mode generate or --mode all.")
    with np.load(path) as npz:
        x = np.asarray(npz["X"], dtype=np.complex64)
        indices = {
            name: np.asarray(npz[f"{name}_indices"], dtype=np.int64)
            for name in ("train", "val", "test")
        }
    expected = (cfg.windows, cfg.samples_per_window, len(PILOT_FRACTIONS))
    if x.shape != expected:
        raise ValueError(f"Dataset {path} has X={x.shape}; current config expects {expected}.")
    return x, indices


def make_loaders(
    x: np.ndarray,
    indices: dict[str, np.ndarray],
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, DataLoader[torch.Tensor]]:
    loaders: dict[str, DataLoader[torch.Tensor]] = {}
    for name in ("train", "val", "test"):
        dataset = PerLinkTokenDataset(
            x,
            indices[name],
            cfg.samples_per_step,
            cfg.gain_scale,
            augment_phase=name == "train",
        )
        generator = torch.Generator()
        generator.manual_seed(cfg.seed + (0 if name == "train" else 1))
        loaders[name] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=name == "train",
            drop_last=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
    return loaders


def train_one_bandwidth(
    cfg: ExperimentConfig,
    output_dir: Path,
    bandwidth_mhz: float,
    device: torch.device,
) -> dict[str, object]:
    x, indices = load_bandwidth_data(cfg, output_dir, bandwidth_mhz)
    loaders = make_loaders(x, indices, cfg, device)
    set_all_seeds(cfg.seed)
    model = SimpleChannelGRU(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"gru_{bandwidth_tag(bandwidth_mhz)}_best.pt"

    best_val_nmse = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    print(f"\n[TRAIN] bandwidth={bandwidth_mhz:g} MHz")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for seq in loaders["train"]:
            seq = seq.to(device, non_blocking=device.type == "cuda")
            loss = training_loss(model, seq, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1

        val = evaluate(model, loaders["val"], cfg, device)
        train_loss = epoch_loss / max(batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_nmse": float(val["nmse"]),
                "val_nmse_db": float(val["nmse_db"]),
            }
        )
        improved = float(val["nmse"]) < best_val_nmse
        if improved:
            best_val_nmse = float(val["nmse"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "bandwidth_mhz": bandwidth_mhz,
                    "val_metrics": val,
                    "config": asdict(cfg),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or improved and epoch % 5 == 0:
            print(
                f"[EPOCH {epoch:03d}] train={train_loss:.4e} "
                f"val_NMSE={float(val['nmse_db']):7.3f} dB "
                f"best={10*math.log10(max(best_val_nmse, 1e-15)):7.3f} dB"
            )
        if epoch >= cfg.min_epochs and stale_epochs >= cfg.patience:
            print(f"[EARLY STOP] epoch={epoch}, best_epoch={best_epoch}")
            break

    history_path = output_dir / f"history_{bandwidth_tag(bandwidth_mhz)}.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(
        f"[TRAIN DONE] bandwidth={bandwidth_mhz:g} MHz, best_epoch={best_epoch}, "
        f"elapsed={(time.perf_counter()-start_time)/60:.2f} min"
    )
    return {
        "bandwidth_mhz": bandwidth_mhz,
        "best_epoch": best_epoch,
        "best_val_nmse": best_val_nmse,
        "checkpoint": str(checkpoint_path),
    }


def test_one_bandwidth(
    cfg: ExperimentConfig,
    output_dir: Path,
    bandwidth_mhz: float,
    device: torch.device,
) -> dict[str, object]:
    x, indices = load_bandwidth_data(cfg, output_dir, bandwidth_mhz)
    loaders = make_loaders(x, indices, cfg, device)
    checkpoint_path = (
        output_dir / "checkpoints" / f"gru_{bandwidth_tag(bandwidth_mhz)}_best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint_path}; run --mode train or all.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SimpleChannelGRU(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics = evaluate(model, loaders["test"], cfg, device)
    result: dict[str, object] = {
        "bandwidth_mhz": bandwidth_mhz,
        "best_epoch": int(checkpoint["epoch"]),
        "val_nmse": float(checkpoint["val_metrics"]["nmse"]),
        "val_nmse_db": float(checkpoint["val_metrics"]["nmse_db"]),
        "train_windows": len(indices["train"]),
        "val_windows": len(indices["val"]),
        "test_windows": len(indices["test"]),
        **metrics,
    }
    print(
        f"[TEST] {bandwidth_mhz:5g} MHz | GRU NMSE={float(metrics['nmse_db']):7.3f} dB "
        f"| persistence={float(metrics['persistence_nmse_db']):7.3f} dB "
        f"| best_epoch={int(checkpoint['epoch'])}"
    )
    return result


def save_results(results: list[dict[str, object]], cfg: ExperimentConfig, output_dir: Path) -> None:
    results = sorted(results, key=lambda row: float(row["bandwidth_mhz"]))
    reference = next(
        (row for row in results if math.isclose(float(row["bandwidth_mhz"]), 1.4)),
        results[0],
    )
    reference_db = float(reference["nmse_db"])
    for row in results:
        delta = float(row["nmse_db"]) - reference_db
        row["delta_nmse_db_vs_reference"] = delta
        row["degradation_class"] = (
            "large" if delta > 3.0 else "moderate" if delta > 1.0 else "small_or_none"
        )

    summary_fields = [
        "bandwidth_mhz",
        "best_epoch",
        "train_windows",
        "val_windows",
        "test_windows",
        "val_nmse",
        "val_nmse_db",
        "nmse",
        "nmse_db",
        "delta_nmse_db_vs_reference",
        "degradation_class",
        "token_mse",
        "persistence_nmse",
        "persistence_nmse_db",
        "sequences",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    per_step_fields = ["bandwidth_mhz", "method", "prediction_step", "horizon_ms", "nmse", "nmse_db"]
    with (output_dir / "per_step.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_step_fields)
        writer.writeheader()
        for row in results:
            for method, key in (("GRU", "per_step_nmse"), ("persistence", "persistence_per_step_nmse")):
                for step, nmse in enumerate(row[key], start=1):
                    writer.writerow(
                        {
                            "bandwidth_mhz": row["bandwidth_mhz"],
                            "method": method,
                            "prediction_step": step,
                            "horizon_ms": step
                            * cfg.samples_per_step
                            * cfg.csi_period_s
                            * 1000.0,
                            "nmse": nmse,
                            "nmse_db": 10.0 * math.log10(max(float(nmse), 1e-15)),
                        }
                    )

    serializable_results = []
    for row in results:
        serializable_results.append(
            {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in row.items()
            }
        )
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": asdict(cfg), "results": serializable_results},
            handle,
            indent=2,
            ensure_ascii=False,
        )

    bandwidths = [float(row["bandwidth_mhz"]) for row in results]
    gru_db = [float(row["nmse_db"]) for row in results]
    persistence_db = [float(row["persistence_nmse_db"]) for row in results]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].plot(bandwidths, gru_db, "o-", linewidth=2, label="GRU")
    axes[0].plot(bandwidths, persistence_db, "s--", linewidth=1.5, label="Persistence")
    axes[0].set_xlabel("Bandwidth (MHz)")
    axes[0].set_ylabel("Weighted 5-step NMSE (dB, lower is better)")
    axes[0].set_xticks(bandwidths)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    step_axis = np.arange(1, cfg.pred_steps + 1)
    for row in results:
        axes[1].plot(
            step_axis,
            row["per_step_nmse_db"],
            marker="o",
            label=f"{float(row['bandwidth_mhz']):g} MHz",
        )
    axes[1].set_xlabel("Prediction step (25 ms each)")
    axes[1].set_ylabel("GRU NMSE (dB)")
    axes[1].set_xticks(step_axis)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.suptitle("Controlled GRU channel-prediction bandwidth sweep")
    figure.tight_layout()
    figure.savefig(output_dir / "bandwidth_nmse.png", dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "train", "test", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("output/bandwidth_gru"))
    parser.add_argument("--bandwidths", type=parse_bandwidths, default=DEFAULT_BANDWIDTHS_MHZ)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--windows", type=int, default=320)
    parser.add_argument("--samples-per-window", type=int, default=100)
    parser.add_argument("--tdl-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--snr-db", type=float, default=30.0)
    parser.add_argument("--delay-spread-ns", type=float, default=30.0)
    parser.add_argument("--speed-mps", type=float, default=0.5)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        bandwidths_mhz=tuple(args.bandwidths),
        carrier_frequency_hz=2.2e9,
        delay_spread_s=args.delay_spread_ns * 1e-9,
        speed_mps=args.speed_mps,
        snr_db=args.snr_db,
        windows=args.windows,
        samples_per_window=args.samples_per_window,
        samples_per_step=5,
        csi_period_s=5e-3,
        tdl_batch_size=args.tdl_batch_size,
        train_fraction=0.8,
        val_fraction=0.1,
        seed=args.seed,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        gain_scale=20.0,
        eps_norm=1e-6,
        pred_steps=5,
        loss_gamma=0.5,
        max_age_feature=32,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=2e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        patience=args.patience,
        min_epochs=args.min_epochs,
    )


def print_final_table(results: Iterable[dict[str, object]]) -> None:
    rows = sorted(results, key=lambda row: float(row["bandwidth_mhz"]))
    print("\n" + "=" * 83)
    print(" bandwidth | GRU NMSE (dB) | delta vs 1.4/ref | persistence (dB) | judgment")
    print("-" * 83)
    for row in rows:
        print(
            f" {float(row['bandwidth_mhz']):8g} | {float(row['nmse_db']):13.3f} | "
            f"{float(row['delta_nmse_db_vs_reference']):17.3f} | "
            f"{float(row['persistence_nmse_db']):16.3f} | {row['degradation_class']}"
        )
    print("=" * 83)


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    _validate_config(cfg)
    device = choose_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2)
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[CUDA] {torch.cuda.get_device_name(device)}")
    print(f"[OUTPUT] {output_dir}")

    if args.mode in ("generate", "all"):
        generate_datasets(cfg, output_dir, device)
    if args.mode in ("train", "all"):
        for bandwidth_mhz in cfg.bandwidths_mhz:
            train_one_bandwidth(cfg, output_dir, bandwidth_mhz, device)
    if args.mode in ("test", "all"):
        results = [
            test_one_bandwidth(cfg, output_dir, bandwidth_mhz, device)
            for bandwidth_mhz in cfg.bandwidths_mhz
        ]
        save_results(results, cfg, output_dir)
        # save_results adds comparison fields in place.
        print_final_table(results)
        print(f"[RESULT] {output_dir / 'summary.csv'}")
        print(f"[PLOT]   {output_dir / 'bandwidth_nmse.png'}")


if __name__ == "__main__":
    main()
