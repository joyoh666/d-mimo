#!/usr/bin/env python3
"""Fixed-15-kHz-SCS wideband GRU channel-prediction experiment.

This standalone experiment corrects the earlier six-pilot sweep.  The
subcarrier spacing stays fixed at 15 kHz and the number of modeled
subcarriers grows with bandwidth::

    N_sc = round(bandwidth / 15 kHz)

The full frequency response at every subcarrier is packed into each 25 ms
gain/cos/sin token and predicted jointly by the same GRU-64 architecture.
Consequently, both the GRU input/output dimension and its operation count grow
with bandwidth.  Underlying TDL-A path coefficients, path delays, and window
splits are shared across bandwidths for a controlled comparison.

The existing project notebooks and the earlier scalar-pilot experiment are not
modified.

Run the complete experiment::

    .venv/bin/python fixed_scs_wideband_gru_experiment.py --mode all

Quick smoke test::

    .venv/bin/python fixed_scs_wideband_gru_experiment.py --mode all \
        --bandwidths 1.4,5 --windows 16 --epochs 2 --min-epochs 1 \
        --output-dir output/fixed_scs_wideband_gru_smoke
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_BANDWIDTHS_MHZ = (1.4, 5.0, 9.0, 15.0)


@dataclass(frozen=True)
class WidebandConfig:
    bandwidths_mhz: tuple[float, ...]
    subcarrier_spacing_hz: float
    carrier_frequency_hz: float
    delay_spread_s: float
    speed_mps: float
    snr_db: float
    channel_gain: float
    windows: int
    samples_per_window: int
    samples_per_step: int
    csi_period_s: float
    tdl_batch_size: int
    num_links: int
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
    eval_every_epochs: int
    patience_evals: int
    min_epochs: int
    benchmark_repeats: int

    @property
    def steps_per_window(self) -> int:
        return self.samples_per_window // self.samples_per_step

    def num_subcarriers(self, bandwidth_mhz: float) -> int:
        return max(1, int(round(bandwidth_mhz * 1e6 / self.subcarrier_spacing_hz)))

    def token_dim(self, bandwidth_mhz: float) -> int:
        return 3 * self.samples_per_step * self.num_subcarriers(bandwidth_mhz)


def parse_bandwidths(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values or any((not np.isfinite(value)) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Bandwidths must be positive MHz values.")
    if len(values) != len(set(values)):
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


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but CUDA is unavailable.")
    return device


def validate_config(cfg: WidebandConfig) -> None:
    if cfg.samples_per_window % cfg.samples_per_step:
        raise ValueError("samples-per-window must be divisible by samples-per-step.")
    if cfg.steps_per_window < cfg.pred_steps + 2:
        raise ValueError("Each window must contain at least pred-steps + 2 tokens.")
    if cfg.subcarrier_spacing_hz <= 0:
        raise ValueError("Subcarrier spacing must be positive.")
    if cfg.windows < 10 or cfg.batch_size < 1 or cfg.epochs < 1:
        raise ValueError("Use at least 10 windows and positive batch-size/epochs.")
    if cfg.channel_gain <= 0 or cfg.delay_spread_s <= 0 or cfg.csi_period_s <= 0:
        raise ValueError("Channel gain, delay spread, and CSI period must be positive.")


def make_splits(cfg: WidebandConfig) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(cfg.seed)
    permutation = generator.permutation(cfg.windows)
    num_train = int(math.floor(cfg.train_fraction * cfg.windows))
    num_val = int(math.floor(cfg.val_fraction * cfg.windows))
    num_test = cfg.windows - num_train - num_val
    if min(num_train, num_val, num_test) < 1:
        raise ValueError(
            f"Empty split: train={num_train}, val={num_val}, test={num_test}."
        )
    return {
        "train": permutation[:num_train],
        "val": permutation[num_train : num_train + num_val],
        "test": permutation[num_train + num_val :],
    }


def centered_frequency_grid(num_subcarriers: int, spacing_hz: float, device: torch.device) -> torch.Tensor:
    """Return an exactly spaced, baseband-centered frequency grid."""
    indices = torch.arange(num_subcarriers, dtype=torch.float64, device=device)
    return (indices - (num_subcarriers - 1) / 2.0) * spacing_hz


def data_path(output_dir: Path, bandwidth_mhz: float, num_subcarriers: int) -> Path:
    return output_dir / "data" / f"channel_{bandwidth_tag(bandwidth_mhz)}_nsc{num_subcarriers}.npy"


def metadata_path(output_dir: Path, bandwidth_mhz: float, num_subcarriers: int) -> Path:
    return output_dir / "data" / f"channel_{bandwidth_tag(bandwidth_mhz)}_nsc{num_subcarriers}.json"


def generate_datasets(cfg: WidebandConfig, output_dir: Path, device: torch.device) -> None:
    try:
        from sionna.phy.channel.tr38901 import TDL
    except ImportError as exc:
        raise RuntimeError("Sionna is required; run with this project's .venv.") from exc

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits = make_splits(cfg)
    memmaps: dict[float, np.memmap] = {}
    noise_generators: dict[float, torch.Generator] = {}
    for index, bandwidth_mhz in enumerate(cfg.bandwidths_mhz):
        num_subcarriers = cfg.num_subcarriers(bandwidth_mhz)
        path = data_path(output_dir, bandwidth_mhz, num_subcarriers)
        memmaps[bandwidth_mhz] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.complex64,
            shape=(cfg.windows, cfg.samples_per_window, cfg.num_links, num_subcarriers),
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.seed + 50_021 + index)
        noise_generators[bandwidth_mhz] = generator

    set_all_seeds(cfg.seed)
    tdl = TDL(
        model="A",
        delay_spread=cfg.delay_spread_s,
        carrier_frequency=cfg.carrier_frequency_hz,
        min_speed=cfg.speed_mps,
        max_speed=cfg.speed_mps,
        num_rx_ant=1,
        num_tx_ant=cfg.num_links,
        device=str(device),
    )
    print(
        f"[GENERATE] fixed SCS={cfg.subcarrier_spacing_hz/1e3:g} kHz, "
        f"TDL-A fc={cfg.carrier_frequency_hz/1e9:g} GHz, "
        f"DS={cfg.delay_spread_s*1e9:g} ns, speed={cfg.speed_mps:g} m/s"
    )
    for bandwidth_mhz in cfg.bandwidths_mhz:
        nsc = cfg.num_subcarriers(bandwidth_mhz)
        print(
            f"[GRID] {bandwidth_mhz:g} MHz -> Nsc={nsc}, "
            f"modeled BW={nsc*cfg.subcarrier_spacing_hz/1e6:.3f} MHz"
        )

    start = 0
    while start < cfg.windows:
        batch_size = min(cfg.tdl_batch_size, cfg.windows - start)
        h, delays = tdl(
            batch_size=batch_size,
            num_time_steps=cfg.samples_per_window,
            sampling_frequency=1.0 / cfg.csi_period_s,
        )
        # h [B,1,1,1,link,path,time], delays [B,1,1,path]
        h = h[:, 0, 0, 0, :, :, :]
        delays = delays[:, 0, 0, :].to(torch.float64)
        if h.ndim != 4 or h.shape[1] != cfg.num_links:
            raise RuntimeError(f"Unexpected TDL coefficient shape {tuple(h.shape)}")

        for bandwidth_mhz in cfg.bandwidths_mhz:
            num_subcarriers = cfg.num_subcarriers(bandwidth_mhz)
            frequencies = centered_frequency_grid(
                num_subcarriers, cfg.subcarrier_spacing_hz, device
            )
            phase = torch.exp(
                -1j * 2.0 * math.pi * frequencies.view(1, -1, 1) * delays.unsqueeze(1)
            ).to(h.dtype)
            # [B,link,path,time] x [B,freq,path] -> [B,time,link,freq]
            cfr = cfg.channel_gain * torch.einsum("bapt,bfp->btaf", h, phase)
            signal_power = cfr.abs().square().mean(dim=(1, 3), keepdim=True)
            noise_power = signal_power / (10.0 ** (cfg.snr_db / 10.0))
            generator = noise_generators[bandwidth_mhz]
            noise = torch.complex(
                torch.randn(cfr.shape, device=device, generator=generator),
                torch.randn(cfr.shape, device=device, generator=generator),
            ) / math.sqrt(2.0)
            observed = cfr + noise * noise_power.clamp_min(1e-15).sqrt()
            memmaps[bandwidth_mhz][start : start + batch_size] = (
                observed.detach().cpu().numpy().astype(np.complex64)
            )
            del cfr, noise, observed, phase

        start += batch_size
        print(f"[GENERATE] {start:4d}/{cfg.windows} matched windows", flush=True)

    for bandwidth_mhz, memmap in memmaps.items():
        memmap.flush()
        num_subcarriers = cfg.num_subcarriers(bandwidth_mhz)
        metadata = {
            "bandwidth_mhz": bandwidth_mhz,
            "subcarrier_spacing_hz": cfg.subcarrier_spacing_hz,
            "num_subcarriers": num_subcarriers,
            "modeled_bandwidth_hz": num_subcarriers * cfg.subcarrier_spacing_hz,
            "shape": list(memmap.shape),
            "dtype": str(memmap.dtype),
            "train_indices": splits["train"].tolist(),
            "val_indices": splits["val"].tolist(),
            "test_indices": splits["test"].tolist(),
            "carrier_frequency_hz": cfg.carrier_frequency_hz,
            "delay_spread_s": cfg.delay_spread_s,
            "speed_mps": cfg.speed_mps,
            "snr_db": cfg.snr_db,
            "channel_gain": cfg.channel_gain,
        }
        path = metadata_path(output_dir, bandwidth_mhz, num_subcarriers)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"[DATA] saved {data_path(output_dir, bandwidth_mhz, num_subcarriers)}")


def complex_to_tokens(channel: torch.Tensor, gain_scale: float) -> torch.Tensor:
    magnitude = channel.abs() * gain_scale
    angle = torch.angle(channel)
    return torch.cat((magnitude, torch.cos(angle), torch.sin(angle)), dim=-1).float()


def postprocess_tokens(raw: torch.Tensor, complex_dim: int, eps_norm: float) -> torch.Tensor:
    gain = F.softplus(raw[..., :complex_dim])
    cos_raw = raw[..., complex_dim : 2 * complex_dim]
    sin_raw = raw[..., 2 * complex_dim : 3 * complex_dim]
    norm = torch.sqrt(cos_raw.square() + sin_raw.square() + eps_norm)
    return torch.cat((gain, cos_raw / norm, sin_raw / norm), dim=-1)


def tokens_to_complex(tokens: torch.Tensor, complex_dim: int, gain_scale: float) -> torch.Tensor:
    gain = tokens[..., :complex_dim] / gain_scale
    cos = tokens[..., complex_dim : 2 * complex_dim]
    sin = tokens[..., 2 * complex_dim : 3 * complex_dim]
    return torch.complex(gain * cos, gain * sin)


class WidebandTokenDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        array: np.ndarray,
        window_indices: np.ndarray,
        cfg: WidebandConfig,
        augment_phase: bool,
    ) -> None:
        if array.ndim != 4 or array.shape[2] != cfg.num_links:
            raise ValueError(f"Expected [window,time,link,subcarrier], got {array.shape}")
        self.array = array
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.cfg = cfg
        self.augment_phase = augment_phase
        self.num_subcarriers = array.shape[-1]
        self.complex_dim = cfg.samples_per_step * self.num_subcarriers

    def __len__(self) -> int:
        return len(self.window_indices) * self.cfg.num_links

    def __getitem__(self, index: int) -> torch.Tensor:
        local_window, link = divmod(index, self.cfg.num_links)
        window = int(self.window_indices[local_window])
        channel = torch.from_numpy(np.array(self.array[window, :, link, :], copy=True))
        if self.augment_phase:
            phase = torch.rand((), dtype=torch.float32) * (2.0 * math.pi)
            channel = channel * torch.polar(torch.ones((), dtype=torch.float32), phase)
        channel = channel.reshape(
            self.cfg.steps_per_window, self.cfg.samples_per_step, self.num_subcarriers
        ).flatten(start_dim=1)
        return complex_to_tokens(channel, self.cfg.gain_scale)


class WidebandChannelGRU(nn.Module):
    def __init__(self, token_dim: int, cfg: WidebandConfig) -> None:
        super().__init__()
        if token_dim % 3:
            raise ValueError("token_dim must be divisible by three.")
        self.token_dim = token_dim
        self.complex_dim = token_dim // 3
        self.max_age_feature = cfg.max_age_feature
        self.eps_norm = cfg.eps_norm
        self.init_token = nn.Parameter(torch.zeros(token_dim))
        nn.init.normal_(self.init_token, std=0.02)
        self.gru = nn.GRU(
            input_size=token_dim + 1,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(cfg.hidden_size, token_dim)

    def step(
        self, token: torch.Tensor, age: int, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_age = min(age, self.max_age_feature) / float(self.max_age_feature)
        age = torch.full(
            (token.shape[0], 1), normalized_age, device=token.device, dtype=token.dtype
        )
        output, hidden = self.gru(torch.cat((token, age), dim=-1).unsqueeze(1), hidden)
        prediction = postprocess_tokens(
            self.output(output[:, 0]), self.complex_dim, self.eps_norm
        )
        return prediction, hidden


def warmup_hidden(model: WidebandChannelGRU, sequence: torch.Tensor, t0: int) -> torch.Tensor:
    batch, _, token_dim = sequence.shape
    initial = model.init_token.view(1, 1, token_dim).expand(batch, 1, token_dim)
    prefix = torch.cat((initial, sequence[:, :t0]), dim=1) if t0 else initial
    ages = torch.zeros((batch, prefix.shape[1], 1), device=sequence.device, dtype=sequence.dtype)
    _, hidden = model.gru(torch.cat((prefix, ages), dim=-1))
    return hidden


def rollout(
    model: WidebandChannelGRU,
    sequence: torch.Tensor,
    t0: int,
    pred_steps: int,
) -> torch.Tensor:
    hidden = warmup_hidden(model, sequence, t0)
    current = sequence[:, t0]
    predictions = []
    for step in range(pred_steps):
        prediction, hidden = model.step(current, step, hidden)
        predictions.append(prediction)
        current = prediction.detach()
    return torch.stack(predictions, dim=1)


def training_loss(
    model: WidebandChannelGRU, sequence: torch.Tensor, cfg: WidebandConfig
) -> torch.Tensor:
    last_t0 = sequence.shape[1] - cfg.pred_steps - 1
    t0 = int(torch.randint(0, last_t0 + 1, (1,), device=sequence.device).item())
    prediction = rollout(model, sequence, t0, cfg.pred_steps)
    target = sequence[:, t0 + 1 : t0 + 1 + cfg.pred_steps]
    weights = torch.tensor(
        [cfg.loss_gamma**step for step in range(cfg.pred_steps)],
        device=sequence.device,
        dtype=sequence.dtype,
    )
    per_step = (prediction - target).square().mean(dim=(0, 2))
    return (per_step * weights).sum() / weights.sum()


@torch.no_grad()
def evaluate(
    model: WidebandChannelGRU,
    loader: DataLoader[torch.Tensor],
    cfg: WidebandConfig,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    error = torch.zeros(cfg.pred_steps, dtype=torch.float64)
    energy = torch.zeros(cfg.pred_steps, dtype=torch.float64)
    persistence_error = torch.zeros(cfg.pred_steps, dtype=torch.float64)
    evaluated_predictions = 0
    for sequence in loader:
        sequence = sequence.to(device, non_blocking=device.type == "cuda")
        for t0 in range(sequence.shape[1] - cfg.pred_steps):
            prediction_tokens = rollout(model, sequence, t0, cfg.pred_steps)
            target_tokens = sequence[:, t0 + 1 : t0 + 1 + cfg.pred_steps]
            prediction = tokens_to_complex(
                prediction_tokens, model.complex_dim, cfg.gain_scale
            )
            target = tokens_to_complex(target_tokens, model.complex_dim, cfg.gain_scale)
            context = tokens_to_complex(
                sequence[:, t0], model.complex_dim, cfg.gain_scale
            ).unsqueeze(1)
            persistence = context.expand(-1, cfg.pred_steps, -1)
            error += (prediction - target).abs().square().sum(dim=(0, 2)).cpu()
            energy += target.abs().square().sum(dim=(0, 2)).cpu()
            persistence_error += (
                (persistence - target).abs().square().sum(dim=(0, 2)).cpu()
            )
            evaluated_predictions += sequence.shape[0]

    weights = torch.tensor(
        [cfg.loss_gamma**step for step in range(cfg.pred_steps)], dtype=torch.float64
    )
    weighted_energy = (weights * energy).sum().clamp_min(1e-15)
    nmse = float(((weights * error).sum() / weighted_energy).item())
    persistence_nmse = float(
        ((weights * persistence_error).sum() / weighted_energy).item()
    )
    per_step = (error / energy.clamp_min(1e-15)).numpy()
    persistence_per_step = (persistence_error / energy.clamp_min(1e-15)).numpy()
    return {
        "nmse": nmse,
        "nmse_db": 10.0 * math.log10(max(nmse, 1e-15)),
        "per_step_nmse": per_step.tolist(),
        "per_step_nmse_db": (
            10.0 * np.log10(np.maximum(per_step, 1e-15))
        ).tolist(),
        "persistence_nmse": persistence_nmse,
        "persistence_nmse_db": 10.0 * math.log10(max(persistence_nmse, 1e-15)),
        "persistence_per_step_nmse": persistence_per_step.tolist(),
        "evaluated_predictions": evaluated_predictions,
    }


def load_data_and_indices(
    cfg: WidebandConfig, output_dir: Path, bandwidth_mhz: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    num_subcarriers = cfg.num_subcarriers(bandwidth_mhz)
    array_path = data_path(output_dir, bandwidth_mhz, num_subcarriers)
    info_path = metadata_path(output_dir, bandwidth_mhz, num_subcarriers)
    if not array_path.is_file() or not info_path.is_file():
        raise FileNotFoundError(
            f"Missing fixed-SCS dataset for {bandwidth_mhz:g} MHz; run --mode generate/all."
        )
    array = np.load(array_path, mmap_mode="r")
    with info_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    indices = {
        name: np.asarray(metadata[f"{name}_indices"], dtype=np.int64)
        for name in ("train", "val", "test")
    }
    expected = (cfg.windows, cfg.samples_per_window, cfg.num_links, num_subcarriers)
    if array.shape != expected:
        raise ValueError(f"Dataset shape {array.shape}; expected {expected}.")
    return array, indices


def make_loaders(
    array: np.ndarray,
    indices: dict[str, np.ndarray],
    cfg: WidebandConfig,
    device: torch.device,
) -> dict[str, DataLoader[torch.Tensor]]:
    result: dict[str, DataLoader[torch.Tensor]] = {}
    for split in ("train", "val", "test"):
        dataset = WidebandTokenDataset(
            array, indices[split], cfg, augment_phase=split == "train"
        )
        generator = torch.Generator()
        generator.manual_seed(cfg.seed + (0 if split == "train" else 1))
        result[split] = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=split == "train",
            drop_last=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
    return result


def checkpoint_path(output_dir: Path, bandwidth_mhz: float) -> Path:
    return output_dir / "checkpoints" / f"wideband_gru_{bandwidth_tag(bandwidth_mhz)}_best.pt"


def train_one_bandwidth(
    cfg: WidebandConfig,
    output_dir: Path,
    bandwidth_mhz: float,
    device: torch.device,
) -> dict[str, object]:
    array, indices = load_data_and_indices(cfg, output_dir, bandwidth_mhz)
    loaders = make_loaders(array, indices, cfg, device)
    token_dim = cfg.token_dim(bandwidth_mhz)
    set_all_seeds(cfg.seed)
    model = WidebandChannelGRU(token_dim, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    output_dir.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(output_dir, bandwidth_mhz)
    best_val_nmse = math.inf
    best_epoch = 0
    stale_evals = 0
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    print(
        f"\n[TRAIN] {bandwidth_mhz:g} MHz, Nsc={cfg.num_subcarriers(bandwidth_mhz)}, "
        f"token_dim={token_dim}, params={sum(p.numel() for p in model.parameters()):,}"
    )
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for sequence in loaders["train"]:
            sequence = sequence.to(device, non_blocking=device.type == "cuda")
            loss = training_loss(model, sequence, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss_sum += float(loss.item())
            batch_count += 1

        train_loss = loss_sum / max(batch_count, 1)
        should_evaluate = epoch == 1 or epoch % cfg.eval_every_epochs == 0 or epoch == cfg.epochs
        if should_evaluate:
            validation = evaluate(model, loaders["val"], cfg, device)
            val_nmse = float(validation["nmse"])
            improved = val_nmse < best_val_nmse
            if improved:
                best_val_nmse = val_nmse
                best_epoch = epoch
                stale_evals = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "bandwidth_mhz": bandwidth_mhz,
                        "num_subcarriers": cfg.num_subcarriers(bandwidth_mhz),
                        "token_dim": token_dim,
                        "val_metrics": validation,
                        "config": asdict(cfg),
                    },
                    path,
                )
            else:
                stale_evals += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_nmse": val_nmse,
                    "val_nmse_db": float(validation["nmse_db"]),
                }
            )
            print(
                f"[EPOCH {epoch:03d}] train={train_loss:.4e} "
                f"val={float(validation['nmse_db']):7.3f} dB "
                f"best={10*math.log10(max(best_val_nmse,1e-15)):7.3f} dB"
            )
            if epoch >= cfg.min_epochs and stale_evals >= cfg.patience_evals:
                print(f"[EARLY STOP] epoch={epoch}, best_epoch={best_epoch}")
                break

    training_seconds = time.perf_counter() - start_time
    history_file = output_dir / f"history_{bandwidth_tag(bandwidth_mhz)}.csv"
    with history_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(
        f"[TRAIN DONE] {bandwidth_mhz:g} MHz, best_epoch={best_epoch}, "
        f"time={training_seconds:.2f} s"
    )
    return {"training_seconds": training_seconds, "best_epoch": best_epoch}


@torch.no_grad()
def benchmark_inference(
    model: WidebandChannelGRU,
    sequence: torch.Tensor,
    cfg: WidebandConfig,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    t0 = (sequence.shape[1] - cfg.pred_steps) // 2
    warmups = 20
    repeats = cfg.benchmark_repeats
    for _ in range(warmups):
        rollout(model, sequence, t0, cfg.pred_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        rollout(model, sequence, t0, cfg.pred_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / repeats
    batch = sequence.shape[0]
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return {
        "benchmark_batch_size": batch,
        "inference_latency_ms_per_batch": latency_ms,
        "inference_throughput_sequences_s": batch * repeats / elapsed,
        "peak_inference_memory_mb": peak_memory / (1024.0**2),
    }


def test_one_bandwidth(
    cfg: WidebandConfig,
    output_dir: Path,
    bandwidth_mhz: float,
    device: torch.device,
    training_seconds: float | None = None,
) -> dict[str, object]:
    array, indices = load_data_and_indices(cfg, output_dir, bandwidth_mhz)
    loaders = make_loaders(array, indices, cfg, device)
    path = checkpoint_path(output_dir, bandwidth_mhz)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}; run --mode train/all.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = WidebandChannelGRU(int(checkpoint["token_dim"]), cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    test_metrics = evaluate(model, loaders["test"], cfg, device)
    benchmark_sequence = next(iter(loaders["test"])).to(
        device, non_blocking=device.type == "cuda"
    )
    benchmark = benchmark_inference(model, benchmark_sequence, cfg, device)
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    result: dict[str, object] = {
        "bandwidth_mhz": bandwidth_mhz,
        "subcarrier_spacing_khz": cfg.subcarrier_spacing_hz / 1e3,
        "num_subcarriers": cfg.num_subcarriers(bandwidth_mhz),
        "modeled_bandwidth_mhz": cfg.num_subcarriers(bandwidth_mhz)
        * cfg.subcarrier_spacing_hz
        / 1e6,
        "token_dim": model.token_dim,
        "num_parameters": num_parameters,
        "best_epoch": int(checkpoint["epoch"]),
        "training_seconds": training_seconds,
        "train_windows": len(indices["train"]),
        "val_windows": len(indices["val"]),
        "test_windows": len(indices["test"]),
        "val_nmse": float(checkpoint["val_metrics"]["nmse"]),
        "val_nmse_db": float(checkpoint["val_metrics"]["nmse_db"]),
        **test_metrics,
        **benchmark,
    }
    print(
        f"[TEST] {bandwidth_mhz:5g} MHz Nsc={result['num_subcarriers']:4d} | "
        f"NMSE={float(result['nmse_db']):7.3f} dB | "
        f"latency={float(result['inference_latency_ms_per_batch']):7.3f} ms "
        f"(batch={result['benchmark_batch_size']})"
    )
    return result


def save_results(
    results: list[dict[str, object]], cfg: WidebandConfig, output_dir: Path
) -> None:
    results.sort(key=lambda row: float(row["bandwidth_mhz"]))
    reference = next(
        (row for row in results if math.isclose(float(row["bandwidth_mhz"]), 1.4)),
        results[0],
    )
    for row in results:
        row["delta_nmse_db_vs_reference"] = float(row["nmse_db"]) - float(
            reference["nmse_db"]
        )
        row["latency_ratio_vs_reference"] = float(
            row["inference_latency_ms_per_batch"]
        ) / float(reference["inference_latency_ms_per_batch"])
        row["parameter_ratio_vs_reference"] = float(row["num_parameters"]) / float(
            reference["num_parameters"]
        )

    summary_fields = [
        "bandwidth_mhz",
        "subcarrier_spacing_khz",
        "num_subcarriers",
        "modeled_bandwidth_mhz",
        "token_dim",
        "num_parameters",
        "parameter_ratio_vs_reference",
        "best_epoch",
        "training_seconds",
        "val_nmse_db",
        "nmse",
        "nmse_db",
        "delta_nmse_db_vs_reference",
        "persistence_nmse_db",
        "benchmark_batch_size",
        "inference_latency_ms_per_batch",
        "inference_throughput_sequences_s",
        "latency_ratio_vs_reference",
        "peak_inference_memory_mb",
        "train_windows",
        "val_windows",
        "test_windows",
        "evaluated_predictions",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    with (output_dir / "per_step.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["bandwidth_mhz", "prediction_step", "horizon_ms", "nmse", "nmse_db"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            for step, nmse in enumerate(row["per_step_nmse"], start=1):
                writer.writerow(
                    {
                        "bandwidth_mhz": row["bandwidth_mhz"],
                        "prediction_step": step,
                        "horizon_ms": step
                        * cfg.samples_per_step
                        * cfg.csi_period_s
                        * 1000.0,
                        "nmse": nmse,
                        "nmse_db": 10.0 * math.log10(max(float(nmse), 1e-15)),
                    }
                )

    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": asdict(cfg), "results": results}, handle, indent=2)

    bandwidths = [float(row["bandwidth_mhz"]) for row in results]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    axes[0].plot(
        bandwidths,
        [float(row["nmse_db"]) for row in results],
        "o-",
        linewidth=2,
    )
    axes[0].set_xlabel("Bandwidth (MHz), SCS fixed at 15 kHz")
    axes[0].set_ylabel("Weighted 5-step NMSE (dB)")
    axes[0].set_title("Prediction accuracy")

    axes[1].plot(
        bandwidths,
        [float(row["inference_latency_ms_per_batch"]) for row in results],
        "s-",
        linewidth=2,
        color="tab:orange",
    )
    axes[1].set_xlabel("Bandwidth (MHz)")
    axes[1].set_ylabel("5-step latency (ms / batch)")
    axes[1].set_title("Measured GPU inference")

    axes[2].plot(
        bandwidths,
        [float(row["num_parameters"]) / 1e6 for row in results],
        "^-",
        linewidth=2,
        color="tab:green",
    )
    axes[2].set_xlabel("Bandwidth (MHz)")
    axes[2].set_ylabel("Trainable parameters (million)")
    axes[2].set_title("Model size")
    for axis in axes:
        axis.set_xticks(bandwidths)
        axis.grid(True, alpha=0.3)
    figure.suptitle("Fixed-SCS full-band GRU sweep")
    figure.tight_layout()
    figure.savefig(output_dir / "fixed_scs_accuracy_compute.png", dpi=180)
    plt.close(figure)


def print_table(results: list[dict[str, object]]) -> None:
    print("\n" + "=" * 113)
    print(
        " BW | Nsc | token dim | params(M) | NMSE(dB) | delta(dB) | "
        "latency(ms/batch) | latency ratio"
    )
    print("-" * 113)
    for row in sorted(results, key=lambda value: float(value["bandwidth_mhz"])):
        print(
            f"{float(row['bandwidth_mhz']):4g} | {int(row['num_subcarriers']):4d} | "
            f"{int(row['token_dim']):9d} | {int(row['num_parameters'])/1e6:9.3f} | "
            f"{float(row['nmse_db']):8.3f} | "
            f"{float(row['delta_nmse_db_vs_reference']):9.3f} | "
            f"{float(row['inference_latency_ms_per_batch']):17.3f} | "
            f"{float(row['latency_ratio_vs_reference']):12.2f}x"
        )
    print("=" * 113)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "train", "test", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("output/fixed_scs_wideband_gru"))
    parser.add_argument("--bandwidths", type=parse_bandwidths, default=DEFAULT_BANDWIDTHS_MHZ)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--samples-per-window", type=int, default=100)
    parser.add_argument("--tdl-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every-epochs", type=int, default=5)
    parser.add_argument("--patience-evals", type=int, default=6)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--snr-db", type=float, default=30.0)
    parser.add_argument("--delay-spread-ns", type=float, default=30.0)
    parser.add_argument("--speed-mps", type=float, default=0.5)
    return parser


def config_from_args(args: argparse.Namespace) -> WidebandConfig:
    return WidebandConfig(
        bandwidths_mhz=tuple(args.bandwidths),
        subcarrier_spacing_hz=15e3,
        carrier_frequency_hz=2.2e9,
        delay_spread_s=args.delay_spread_ns * 1e-9,
        speed_mps=args.speed_mps,
        snr_db=args.snr_db,
        channel_gain=0.05,
        windows=args.windows,
        samples_per_window=args.samples_per_window,
        samples_per_step=5,
        csi_period_s=5e-3,
        tdl_batch_size=args.tdl_batch_size,
        num_links=6,
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
        eval_every_epochs=args.eval_every_epochs,
        patience_evals=args.patience_evals,
        min_epochs=args.min_epochs,
        benchmark_repeats=args.benchmark_repeats,
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    validate_config(cfg)
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

    training_times: dict[float, float] = {}
    if args.mode in ("train", "all"):
        for bandwidth_mhz in cfg.bandwidths_mhz:
            train_info = train_one_bandwidth(cfg, output_dir, bandwidth_mhz, device)
            training_times[bandwidth_mhz] = float(train_info["training_seconds"])

    if args.mode in ("test", "all"):
        results = [
            test_one_bandwidth(
                cfg,
                output_dir,
                bandwidth_mhz,
                device,
                training_times.get(bandwidth_mhz),
            )
            for bandwidth_mhz in cfg.bandwidths_mhz
        ]
        save_results(results, cfg, output_dir)
        print_table(results)
        print(f"[SUMMARY] {output_dir / 'summary.csv'}")
        print(f"[PLOT] {output_dir / 'fixed_scs_accuracy_compute.png'}")


if __name__ == "__main__":
    main()
