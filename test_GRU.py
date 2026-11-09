#!/usr/bin/env python3
"""Evaluate a trained GRU channel predictor on the prepared test split.

This script does not generate or split data. It loads
``predictor/test/X.npz`` directly, expands every window into one sequence per
antenna/channel, and evaluates the trained model against linear-regression and
constant-velocity Kalman-filter baselines.

The model architecture and token settings are reconstructed from the training
checkpoint, so this file stays compatible with ``train_GRU.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_TEST_NPZ = Path(
    "./data_ready/dmimo_3tx_s66_s75/predictor/test/X.npz"
)
DEFAULT_CHECKPOINT_ROOT = Path("./checkpoints_rnn")


def get_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"gpu_id={gpu_id} is invalid; visible CUDA devices are "
            f"0..{torch.cuda.device_count() - 1}"
        )
    return torch.device(f"cuda:{gpu_id}")


def load_checkpoint(path: Path) -> Dict:
    # PyTorch 2.6+ defaults to weights_only=True. The training checkpoint also
    # contains ordinary Python configuration values, so load it explicitly.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain a 'model' state dict: {path}")
    return checkpoint


def find_checkpoint(checkpoint: Optional[Path], root: Path) -> Path:
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint

    best = root / "best_validation_nmse.pt"
    if best.is_file():
        return best

    candidates = sorted(
        root.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found. Expected {best} or another .pt under {root}"
        )
    return candidates[0]


def complex_to_gcs_tokens(
    h_taps: torch.Tensor, gain_scale: float
) -> torch.Tensor:
    """(..., taps) complex -> (..., 3*taps) [scaled gain, cos, sin]."""
    magnitude = h_taps.abs() * float(gain_scale)
    angle = torch.angle(h_taps)
    return torch.cat(
        [magnitude, torch.cos(angle), torch.sin(angle)], dim=-1
    ).to(torch.float32)


def postprocess_pred_tokens(
    tokens: torch.Tensor, taps: int, eps_norm: float
) -> torch.Tensor:
    gain_raw = tokens[..., :taps]
    cos_raw = tokens[..., taps : 2 * taps]
    sin_raw = tokens[..., 2 * taps : 3 * taps]
    gain = F.softplus(gain_raw)
    denominator = torch.sqrt(
        cos_raw * cos_raw + sin_raw * sin_raw + float(eps_norm)
    )
    return torch.cat(
        [gain, cos_raw / denominator, sin_raw / denominator], dim=-1
    )


def tokens_to_complex(
    tokens: torch.Tensor, taps: int, gain_scale: float
) -> torch.Tensor:
    scaled_gain = tokens[..., :taps]
    cosine = tokens[..., taps : 2 * taps]
    sine = tokens[..., 2 * taps : 3 * taps]
    gain = scaled_gain / float(gain_scale)
    return torch.complex(gain * cosine, gain * sine)


class XNPZWindows(Dataset):
    """Prepared complex windows with dynamic channel count C: (N, L, C)."""

    def __init__(self, path: Path, samples_per_step: int):
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        self.samples_per_step = int(samples_per_step)
        self._npz = np.load(path, mmap_mode="r", allow_pickle=False)
        if "X" not in self._npz:
            raise KeyError(f"'X' not found in {path}")
        self.X = self._npz["X"]
        if self.X.ndim != 3 or self.X.shape[-1] < 1:
            raise ValueError(f"Expected X shape (N,L,C), got {self.X.shape}")
        if not np.issubdtype(self.X.dtype, np.complexfloating):
            raise TypeError(f"Expected complex CSI, got {self.X.dtype}")

        self.num_antennas = int(self.X.shape[-1])
        if "num_antennas" in self._npz:
            metadata_count = int(self._npz["num_antennas"])
            if metadata_count != self.num_antennas:
                raise ValueError(
                    f"num_antennas metadata={metadata_count}, but X has "
                    f"C={self.num_antennas}"
                )
        self.dt = float(self._npz["dt"]) if "dt" in self._npz else 0.005
        self.scenario_ids = (
            [int(x) for x in np.asarray(self._npz["scenario_ids"]).reshape(-1)]
            if "scenario_ids" in self._npz
            else []
        )
        length = int(self.X.shape[1])
        if length % self.samples_per_step != 0:
            raise ValueError(
                f"L={length} is not divisible by "
                f"samples_per_step={self.samples_per_step}"
            )

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        # A writable copy avoids PyTorch warnings caused by read-only NPZ arrays.
        return torch.from_numpy(np.array(self.X[int(index)], copy=True))


class PerAntennaSequenceDataset(Dataset):
    """One GCS-token sequence for each (window, antenna) pair."""

    def __init__(self, windows: XNPZWindows, gain_scale: float):
        self.windows = windows
        self.gain_scale = float(gain_scale)
        self.num_antennas = windows.num_antennas
        self.sequence_length = int(windows.X.shape[1]) // windows.samples_per_step

    def __len__(self) -> int:
        return len(self.windows) * self.num_antennas

    def item_location(self, index: int) -> Tuple[int, int]:
        return index // self.num_antennas, index % self.num_antennas

    def __getitem__(self, index: int) -> torch.Tensor:
        window_index, antenna_index = self.item_location(int(index))
        channel = self.windows[window_index][:, antenna_index]
        taps = channel.reshape(
            self.sequence_length, self.windows.samples_per_step
        )
        return complex_to_gcs_tokens(taps, self.gain_scale)


class SimpleChannelRNN(nn.Module):
    """Exact inference-time architecture used by train_GRU.py."""

    def __init__(
        self,
        tok_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        rnn_type: str,
        use_age_feature: bool,
    ):
        super().__init__()
        self.tok_dim = int(tok_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.rnn_type = str(rnn_type).upper()
        self.use_age_feature = bool(use_age_feature)
        self.in_dim = self.tok_dim + int(self.use_age_feature)
        self.init_tok = nn.Parameter(torch.zeros(self.tok_dim))

        rnn_kwargs = dict(
            input_size=self.in_dim,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            dropout=(float(dropout) if self.layers > 1 else 0.0),
        )
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(**rnn_kwargs)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(nonlinearity="tanh", **rnn_kwargs)
        else:
            raise ValueError(f"Unknown rnn_type={rnn_type}")
        self.out_proj = nn.Linear(self.hidden, self.tok_dim)

    def rnn_step(
        self,
        token: torch.Tensor,
        age_norm: Optional[torch.Tensor],
        hidden: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_age_feature:
            if age_norm is None:
                raise ValueError("age_norm is required when use_age_feature=True")
            token = torch.cat([token, age_norm], dim=-1)
        output, next_hidden = self.rnn(token.unsqueeze(1), hidden)
        return output[:, 0, :], next_hidden


def build_model(checkpoint: Dict, tok_dim: int) -> SimpleChannelRNN:
    config = checkpoint.get("config", {})
    model = SimpleChannelRNN(
        tok_dim=int(config.get("tok_dim", tok_dim)),
        hidden=int(config.get("hidden_size", 64)),
        layers=int(config.get("num_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
        rnn_type=str(config.get("rnn_type", "GRU")),
        use_age_feature=bool(config.get("use_age_feature", True)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


@torch.no_grad()
def model_kstep_predict(
    model: SimpleChannelRNN,
    sequence: torch.Tensor,
    t0: int,
    pred_steps: int,
    taps: int,
    eps_norm: float,
    max_age_feature: int,
) -> torch.Tensor:
    """Return postprocessed model tokens with shape (B, K, 3*taps)."""
    batch, sequence_length, token_dim = sequence.shape
    if not 0 <= t0 < sequence_length - pred_steps:
        raise ValueError(
            f"t0={t0} is invalid for S={sequence_length}, K={pred_steps}"
        )

    init_token = model.init_tok.view(1, 1, token_dim).expand(
        batch, 1, token_dim
    )
    if model.use_age_feature:
        init_input = torch.cat(
            [
                init_token,
                torch.zeros(
                    (batch, 1, 1),
                    device=sequence.device,
                    dtype=sequence.dtype,
                ),
            ],
            dim=-1,
        )
    else:
        init_input = init_token

    if t0 > 0:
        prefix = sequence[:, :t0, :]
        if model.use_age_feature:
            prefix = torch.cat(
                [
                    prefix,
                    torch.zeros(
                        (batch, t0, 1),
                        device=sequence.device,
                        dtype=sequence.dtype,
                    ),
                ],
                dim=-1,
            )
        prefix = torch.cat([init_input, prefix], dim=1)
    else:
        prefix = init_input
    _, hidden = model.rnn(prefix)

    token = sequence[:, t0, :]
    predictions = []
    for step in range(pred_steps):
        if model.use_age_feature:
            clipped_age = min(step, max_age_feature)
            age_norm = torch.full(
                (batch, 1),
                clipped_age / float(max_age_feature),
                device=sequence.device,
                dtype=sequence.dtype,
            )
        else:
            age_norm = None
        output, hidden = model.rnn_step(token, age_norm, hidden)
        prediction = postprocess_pred_tokens(
            model.out_proj(output), taps=taps, eps_norm=eps_norm
        )
        predictions.append(prediction)
        token = prediction
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def linear_regression_predict(
    context: torch.Tensor, future_length: int
) -> torch.Tensor:
    """Fit complex linear trends to one GT step and extrapolate."""
    _, taps = context.shape
    device = context.device
    time = torch.arange(taps, device=device, dtype=torch.float32)
    time_mean = time.mean()
    time_variance = ((time - time_mean) ** 2).mean().clamp_min(1e-12)

    def extrapolate(values: torch.Tensor) -> torch.Tensor:
        value_mean = values.mean(dim=1, keepdim=True)
        covariance = (
            (time.view(1, -1) - time_mean) * (values - value_mean)
        ).mean(dim=1, keepdim=True)
        slope = covariance / time_variance
        intercept = value_mean - slope * time_mean
        future_time = torch.arange(
            taps,
            taps + future_length,
            device=device,
            dtype=torch.float32,
        ).view(1, -1)
        return slope * future_time + intercept

    return torch.complex(
        extrapolate(context.real.float()),
        extrapolate(context.imag.float()),
    )


@torch.no_grad()
def kalman_predict_1d(
    observations: torch.Tensor,
    future_length: int,
    r_relative: float,
    q_relative: float,
) -> torch.Tensor:
    """Batched constant-velocity Kalman filter and future extrapolation."""
    batch, taps = observations.shape
    device, dtype = observations.device, observations.dtype
    variance = observations.var(dim=1, unbiased=False).clamp_min(1e-12)
    measurement_noise = (float(r_relative) * variance).clamp_min(1e-12)
    process_noise = (float(q_relative) * variance).clamp_min(1e-12)

    state = torch.zeros((batch, 2), device=device, dtype=dtype)
    state[:, 0] = observations[:, 0]
    covariance = torch.eye(2, device=device, dtype=dtype).expand(
        batch, 2, 2
    ).clone()
    transition = torch.tensor(
        [[1.0, 1.0], [0.0, 1.0]], device=device, dtype=dtype
    ).expand(batch, 2, 2)
    observation = torch.tensor(
        [1.0, 0.0], device=device, dtype=dtype
    ).view(1, 1, 2).expand(batch, 1, 2)
    identity = torch.eye(2, device=device, dtype=dtype).expand(batch, 2, 2)
    q_base = torch.tensor(
        [[0.25, 0.5], [0.5, 1.0]], device=device, dtype=dtype
    ).expand(batch, 2, 2)
    process_covariance = process_noise[:, None, None] * q_base

    for step in range(taps):
        state = torch.bmm(transition, state.unsqueeze(-1)).squeeze(-1)
        covariance = (
            torch.bmm(
                torch.bmm(transition, covariance),
                transition.transpose(1, 2),
            )
            + process_covariance
        )
        innovation = observations[:, step] - state[:, 0]
        innovation_variance = (
            covariance[:, 0, 0] + measurement_noise
        ).clamp_min(1e-12)
        gain = covariance[:, :, 0] / innovation_variance[:, None]
        state = state + gain * innovation[:, None]
        gain_observation = torch.bmm(gain.unsqueeze(-1), observation)
        covariance = torch.bmm(identity - gain_observation, covariance)

    predictions = []
    for _ in range(future_length):
        state = torch.bmm(transition, state.unsqueeze(-1)).squeeze(-1)
        predictions.append(state[:, 0])
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def kalman_predict_complex(
    context: torch.Tensor,
    future_length: int,
    r_relative: float,
    q_relative: float,
) -> torch.Tensor:
    return torch.complex(
        kalman_predict_1d(
            context.real.float(), future_length, r_relative, q_relative
        ),
        kalman_predict_1d(
            context.imag.float(), future_length, r_relative, q_relative
        ),
    )


def select_origins(
    sequence_length: int, pred_steps: int, requested_count: int
) -> List[int]:
    """Choose all or evenly spaced valid t0 positions deterministically."""
    count = sequence_length - pred_steps
    if count <= 0:
        raise ValueError(
            f"Sequence length S={sequence_length} must exceed K={pred_steps}"
        )
    if requested_count <= 0 or requested_count >= count:
        return list(range(count))
    values = np.linspace(0, count - 1, requested_count)
    return sorted(set(int(round(x)) for x in values))


def empty_accumulator(pred_steps: int) -> Dict:
    return {
        "token_error": 0.0,
        "token_weight": 0.0,
        "complex_error": 0.0,
        "complex_weight": 0.0,
        "target_power": 0.0,
        "phase_error": 0.0,
        "phase_weight": 0.0,
        "step_error": np.zeros(pred_steps, dtype=np.float64),
        "step_target": np.zeros(pred_steps, dtype=np.float64),
        "evaluated_sequences": 0,
    }


@torch.no_grad()
def evaluate_all_methods(
    model: SimpleChannelRNN,
    loader: DataLoader,
    device: torch.device,
    origins: Sequence[int],
    taps: int,
    gain_scale: float,
    eps_norm: float,
    pred_steps: int,
    loss_gamma: float,
    max_age_feature: int,
    kf_r_relative: float,
    kf_q_relative: float,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray], np.ndarray]:
    model.eval()
    weights = torch.tensor(
        [float(loss_gamma) ** i for i in range(pred_steps)],
        device=device,
        dtype=torch.float32,
    )
    accumulators = {
        method: empty_accumulator(pred_steps)
        for method in ("MODEL", "LR", "KF")
    }
    evm_db_samples: Dict[str, List[np.ndarray]] = {
        method: [] for method in ("MODEL", "LR", "KF")
    }
    reference_gain_samples: List[np.ndarray] = []

    def accumulate(
        method: str,
        predicted_tokens: torch.Tensor,
        predicted_complex: torch.Tensor,
        target_tokens: torch.Tensor,
        target_complex: torch.Tensor,
    ) -> None:
        batch = int(target_tokens.shape[0])
        token_dim = int(target_tokens.shape[-1])
        weighted_steps = weights.view(1, pred_steps, 1)
        token_error = (predicted_tokens - target_tokens) ** 2
        complex_error = (predicted_complex - target_complex).abs() ** 2
        target_power = target_complex.abs() ** 2
        phase_delta = torch.angle(predicted_complex) - torch.angle(target_complex)
        phase_delta = torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta)).abs()

        # One EVM value per (sequence, prediction origin). This is the usual
        # RMS-normalized block EVM: sqrt(sum|prediction-GT|^2 / sum|GT|^2).
        block_error = complex_error.sum(dim=(1, 2))
        block_reference = target_power.sum(dim=(1, 2)).clamp_min(1e-20)
        block_evm = torch.sqrt(block_error / block_reference).clamp_min(1e-12)
        evm_db_samples[method].append(
            (20.0 * torch.log10(block_evm)).detach().cpu().numpy()
        )

        acc = accumulators[method]
        acc["token_error"] += float((token_error * weighted_steps).sum().item())
        acc["token_weight"] += batch * token_dim * float(weights.sum().item())
        acc["complex_error"] += float(
            (complex_error * weighted_steps).sum().item()
        )
        acc["target_power"] += float(
            (target_power * weighted_steps).sum().item()
        )
        acc["complex_weight"] += batch * taps * float(weights.sum().item())
        acc["phase_error"] += float((phase_delta * weighted_steps).sum().item())
        acc["phase_weight"] += batch * taps * float(weights.sum().item())
        acc["step_error"] += (
            complex_error.sum(dim=(0, 2)).detach().cpu().numpy()
        )
        acc["step_target"] += (
            target_power.sum(dim=(0, 2)).detach().cpu().numpy()
        )
        acc["evaluated_sequences"] += batch

    for batch_index, sequence in enumerate(loader, start=1):
        sequence = sequence.to(device, non_blocking=True)
        batch = int(sequence.shape[0])
        for t0 in origins:
            target_tokens = sequence[:, t0 + 1 : t0 + 1 + pred_steps, :]
            target_complex = tokens_to_complex(
                target_tokens, taps=taps, gain_scale=gain_scale
            )
            reference_gain_samples.append(
                torch.sqrt((target_complex.abs() ** 2).mean(dim=(1, 2)))
                .detach()
                .cpu()
                .numpy()
            )
            context_complex = tokens_to_complex(
                sequence[:, t0, :], taps=taps, gain_scale=gain_scale
            )

            model_tokens = model_kstep_predict(
                model,
                sequence,
                t0,
                pred_steps,
                taps,
                eps_norm,
                max_age_feature,
            )
            model_complex = tokens_to_complex(
                model_tokens, taps=taps, gain_scale=gain_scale
            )

            future_length = pred_steps * taps
            lr_complex = linear_regression_predict(
                context_complex, future_length
            ).reshape(batch, pred_steps, taps)
            lr_tokens = complex_to_gcs_tokens(lr_complex, gain_scale)

            kf_complex = kalman_predict_complex(
                context_complex,
                future_length,
                kf_r_relative,
                kf_q_relative,
            ).reshape(batch, pred_steps, taps)
            kf_tokens = complex_to_gcs_tokens(kf_complex, gain_scale)

            accumulate(
                "MODEL",
                model_tokens,
                model_complex,
                target_tokens,
                target_complex,
            )
            accumulate(
                "LR", lr_tokens, lr_complex, target_tokens, target_complex
            )
            accumulate(
                "KF", kf_tokens, kf_complex, target_tokens, target_complex
            )
        print(
            f"[EVAL] batch {batch_index}/{len(loader)} | "
            f"sequences={batch} | origins={len(origins)}"
        )

    results: Dict[str, Dict] = {}
    for method, acc in accumulators.items():
        results[method] = {
            "token_mse_weighted": acc["token_error"]
            / max(acc["token_weight"], 1e-12),
            "complex_mse_weighted": acc["complex_error"]
            / max(acc["complex_weight"], 1e-12),
            "nmse_weighted": acc["complex_error"]
            / max(acc["target_power"], 1e-12),
            "phase_mae_deg_weighted": (
                acc["phase_error"] / max(acc["phase_weight"], 1e-12)
            )
            * (180.0 / math.pi),
            "per_step_nmse": (
                acc["step_error"] / np.maximum(acc["step_target"], 1e-12)
            ).tolist(),
            "evaluated_sequence_origins": int(acc["evaluated_sequences"]),
        }
    evm_arrays = {
        method: np.concatenate(chunks).astype(np.float64, copy=False)
        for method, chunks in evm_db_samples.items()
    }
    reference_gain = np.concatenate(reference_gain_samples).astype(
        np.float64, copy=False
    )
    expected_count = len(reference_gain)
    for method, values in evm_arrays.items():
        if len(values) != expected_count:
            raise RuntimeError(
                f"EVM sample count mismatch for {method}: "
                f"{len(values)} != {expected_count}"
            )
    return results, evm_arrays, reference_gain


@torch.no_grad()
def make_micro_example(
    model: SimpleChannelRNN,
    sequence: torch.Tensor,
    t0: int,
    device: torch.device,
    taps: int,
    gain_scale: float,
    eps_norm: float,
    pred_steps: int,
    max_age_feature: int,
    kf_r_relative: float,
    kf_q_relative: float,
) -> Dict[str, torch.Tensor]:
    sequence = sequence.to(device).unsqueeze(0)
    t0 = max(0, min(int(t0), sequence.shape[1] - pred_steps - 1))
    full_ground_truth = tokens_to_complex(
        sequence, taps, gain_scale
    )[0].reshape(-1)
    forecast_start = (t0 + 1) * taps
    context = tokens_to_complex(
        sequence[:, t0, :], taps, gain_scale
    )[0]
    target = tokens_to_complex(
        sequence[:, t0 + 1 : t0 + 1 + pred_steps, :],
        taps,
        gain_scale,
    )[0]
    model_tokens = model_kstep_predict(
        model,
        sequence,
        t0,
        pred_steps,
        taps,
        eps_norm,
        max_age_feature,
    )
    model_prediction = tokens_to_complex(
        model_tokens, taps, gain_scale
    )[0]
    future_length = pred_steps * taps
    lr_prediction = linear_regression_predict(
        context.unsqueeze(0), future_length
    )[0].reshape(pred_steps, taps)
    kf_prediction = kalman_predict_complex(
        context.unsqueeze(0),
        future_length,
        kf_r_relative,
        kf_q_relative,
    )[0].reshape(pred_steps, taps)
    return {
        "t0": torch.tensor(t0),
        "forecast_start": torch.tensor(forecast_start),
        "full_ground_truth": full_ground_truth.cpu(),
        "context": context.cpu(),
        "target": target.cpu(),
        "model": model_prediction.cpu(),
        "lr": lr_prediction.cpu(),
        "kf": kf_prediction.cpu(),
    }


def print_micro_example(
    pack: Dict[str, torch.Tensor], sample_ms: float
) -> None:
    t0 = int(pack["t0"].item())
    context = pack["context"]
    target = pack["target"]
    model = pack["model"]
    lr = pack["lr"]
    kf = pack["kf"]
    pred_steps, taps = target.shape
    step_ms = taps * sample_ms

    print("\n[MICRO EXAMPLE]")
    print(
        f"t0={t0}, sample={sample_ms:.3f}ms, step={step_ms:.3f}ms, "
        f"horizon={pred_steps * step_ms:.3f}ms"
    )
    print(
        "time_ms  GT_gain  MODEL_gain  LR_gain  KF_gain  "
        "GT_phase  MODEL_phase  |err_MODEL|"
    )
    for step in range(pred_steps):
        for tap in range(taps):
            relative_ms = (step + 1) * step_ms + tap * sample_ms
            gt_value = target[step, tap]
            model_value = model[step, tap]
            print(
                f"{relative_ms:7.2f}  "
                f"{gt_value.abs().item():7.4f}  "
                f"{model_value.abs().item():10.4f}  "
                f"{lr[step, tap].abs().item():7.4f}  "
                f"{kf[step, tap].abs().item():7.4f}  "
                f"{torch.angle(gt_value).item() * 180 / math.pi:8.2f}  "
                f"{torch.angle(model_value).item() * 180 / math.pi:11.2f}  "
                f"{(model_value - gt_value).abs().item():11.4f}"
            )

    target_flat = target.reshape(-1)
    target_power = float((target_flat.abs() ** 2).sum().item())
    for name, prediction in (("MODEL", model), ("LR", lr), ("KF", kf)):
        error = float(((prediction.reshape(-1) - target_flat).abs() ** 2).sum().item())
        print(f"[MICRO {name}] NMSE={error / max(target_power, 1e-12):.6e}")


def empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return values, values
    probabilities = np.arange(1, values.size + 1, dtype=np.float64) / values.size
    return values, probabilities


def align_phase_to_reference(
    prediction_deg: np.ndarray, reference_first_deg: float
) -> np.ndarray:
    """Unwrap a predicted phase and remove an arbitrary 360-degree offset."""
    prediction_deg = np.unwrap(np.deg2rad(prediction_deg)) * 180.0 / math.pi
    offset_turns = round((reference_first_deg - prediction_deg[0]) / 360.0)
    return prediction_deg + 360.0 * offset_turns


def plot_prediction_and_evm_cdf(
    micro: Dict[str, torch.Tensor],
    evm_db: Dict[str, np.ndarray],
    reference_gain: np.ndarray,
    sample_ms: float,
    output_path: Path,
    history_ms: float,
    tail_ms: float,
    cdf_min_db: float,
    cdf_max_db: float,
) -> Dict[str, float]:
    """Create a publication-style prediction trace and EVM CDF figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plots. Install it in the container "
            "or run with an NGC PyTorch image that includes matplotlib."
        ) from exc

    ground_truth = micro["full_ground_truth"].numpy()
    forecast_start = int(micro["forecast_start"].item())
    model_prediction = micro["model"].reshape(-1).numpy()
    lr_prediction = micro["lr"].reshape(-1).numpy()
    kf_prediction = micro["kf"].reshape(-1).numpy()
    forecast_length = len(model_prediction)

    history_samples = max(1, int(round(history_ms / sample_ms)))
    tail_samples = max(0, int(round(tail_ms / sample_ms)))
    plot_start = max(0, forecast_start - history_samples)
    plot_end = min(
        len(ground_truth), forecast_start + forecast_length + tail_samples
    )
    gt_plot = ground_truth[plot_start:plot_end]
    gt_time = np.arange(plot_start, plot_end) * sample_ms
    prediction_time = (
        np.arange(forecast_start, forecast_start + forecast_length) * sample_ms
    )

    gt_gain = np.abs(gt_plot)
    gt_phase = np.unwrap(np.angle(gt_plot)) * 180.0 / math.pi
    forecast_offset = forecast_start - plot_start
    reference_first_phase = float(gt_phase[forecast_offset])

    prediction_series = {
        "GRU (proposed)": (model_prediction, "red", "-."),
        "Linear extrapol.": (lr_prediction, "#0072BD", "--"),
        "Kalman CV": (kf_prediction, "#159447", "--"),
    }

    figure = plt.figure(figsize=(13.5, 5.5))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.12, 1.0), hspace=0.08, wspace=0.26
    )
    gain_axis = figure.add_subplot(grid[0, 0])
    phase_axis = figure.add_subplot(grid[1, 0], sharex=gain_axis)
    cdf_axis = figure.add_subplot(grid[:, 1])

    gain_axis.plot(gt_time, gt_gain, color="black", linewidth=1.8, label="Ground truth")
    phase_axis.plot(gt_time, gt_phase, color="black", linewidth=1.8)
    for label, (prediction, color, line_style) in prediction_series.items():
        gain_axis.plot(
            prediction_time,
            np.abs(prediction),
            color=color,
            linestyle=line_style,
            linewidth=2.2,
            label=label,
        )
        raw_phase = np.angle(prediction) * 180.0 / math.pi
        aligned_phase = align_phase_to_reference(raw_phase, reference_first_phase)
        phase_axis.plot(
            prediction_time,
            aligned_phase,
            color=color,
            linestyle=line_style,
            linewidth=2.2,
        )

    forecast_time = forecast_start * sample_ms
    for axis in (gain_axis, phase_axis):
        axis.axvline(forecast_time, color="0.55", linewidth=1.0, linestyle=":")
        axis.grid(True, alpha=0.22)
    gain_axis.set_ylabel("Gain")
    phase_axis.set_ylabel("Phase (deg)")
    phase_axis.set_xlabel("Time (ms)")
    gain_axis.tick_params(labelbottom=False)
    gain_axis.legend(loc="best", ncol=2, fontsize=9)

    finite_gain = reference_gain[np.isfinite(reference_gain)]
    if finite_gain.size == 0:
        raise ValueError("No finite reference-gain samples are available for the CDF")
    gain_threshold = float(np.quantile(finite_gain, 0.25))
    top_gain_mask = reference_gain >= gain_threshold
    cdf_styles = {
        "MODEL": ("GRU (proposed)", "red"),
        "LR": ("Linear extrapol.", "#0072BD"),
        "KF": ("Kalman CV", "#159447"),
    }
    for method, (label, color) in cdf_styles.items():
        all_x, all_y = empirical_cdf(evm_db[method])
        top_x, top_y = empirical_cdf(evm_db[method][top_gain_mask])
        cdf_axis.plot(
            all_x, all_y, color=color, linewidth=2.0, label=f"{label} (All)"
        )
        cdf_axis.plot(
            top_x,
            top_y,
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"{label} (Top 75% gain)",
        )
    cdf_axis.set_xlabel("EVM (dB)")
    cdf_axis.set_ylabel("Empirical CDF")
    cdf_axis.set_xlim(cdf_min_db, cdf_max_db)
    cdf_axis.set_ylim(0.0, 1.0)
    cdf_axis.grid(True, alpha=0.22)
    cdf_axis.legend(loc="lower right", fontsize=8.5)

    gain_axis.text(0.5, -1.28, "(a)", transform=gain_axis.transAxes, ha="center", fontsize=12)
    cdf_axis.text(0.5, -0.17, "(b)", transform=cdf_axis.transAxes, ha="center", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return {
        "top_75_gain_threshold": gain_threshold,
        "all_sample_count": int(len(reference_gain)),
        "top_75_sample_count": int(np.count_nonzero(top_gain_mask)),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--num-origins",
        type=int,
        default=0,
        help="0 evaluates all valid prediction origins; positive values use evenly spaced origins",
    )
    parser.add_argument("--kf-r-relative", type=float, default=1e-2)
    parser.add_argument("--kf-q-relative", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-micro", action="store_true")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--example-t0", type=int, default=130)
    parser.add_argument(
        "--plot-output", type=Path, default=Path("./test_GRU_evm_cdf.png")
    )
    parser.add_argument(
        "--evm-samples-output",
        type=Path,
        default=Path("./test_GRU_evm_samples.npz"),
    )
    parser.add_argument("--plot-history-ms", type=float, default=300.0)
    parser.add_argument("--plot-tail-ms", type=float, default=100.0)
    parser.add_argument("--cdf-min-db", type=float, default=-30.0)
    parser.add_argument("--cdf-max-db", type=float, default=20.0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--output-json", type=Path, default=Path("./test_GRU_results.json")
    )
    return parser.parse_args(argv)


def print_method(name: str, metrics: Dict, pred_steps: int) -> None:
    print(f"\n[{name}] K={pred_steps}")
    print(f"  token MSE (weighted)   : {metrics['token_mse_weighted']:.6e}")
    print(f"  complex MSE (weighted) : {metrics['complex_mse_weighted']:.6e}")
    print(f"  NMSE (weighted)        : {metrics['nmse_weighted']:.6e}")
    print(
        f"  phase MAE (weighted)   : "
        f"{metrics['phase_mae_deg_weighted']:.3f} deg"
    )
    print(
        "  per-step NMSE          : "
        + "  ".join(f"{x:.3e}" for x in metrics["per_step_nmse"])
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, Dict]:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.gpu_id)
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[CUDA] {torch.cuda.get_device_name(device)}")

    checkpoint_path = find_checkpoint(args.checkpoint, args.checkpoint_root)
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config", {})
    print(f"[CHECKPOINT] {checkpoint_path.resolve()}")
    print(
        f"[CHECKPOINT] epoch={checkpoint.get('epoch', 'unknown')} "
        f"step={checkpoint.get('global_step', 'unknown')}"
    )

    samples_per_step = int(config.get("samples_per_step", 5))
    taps = samples_per_step
    token_dim = int(config.get("tok_dim", 3 * taps))
    gain_scale = float(config.get("gain_scale", 20.0))
    eps_norm = float(config.get("eps_norm", 1e-6))
    pred_steps = int(config.get("pred_steps", 5))
    loss_gamma = float(config.get("loss_gamma", 0.5))
    max_age_feature = int(config.get("max_age_feature", 32))
    if token_dim != 3 * taps:
        raise ValueError(
            f"Checkpoint tok_dim={token_dim}, expected 3*samples_per_step={3 * taps}"
        )
    if max_age_feature <= 0:
        raise ValueError(f"max_age_feature must be positive, got {max_age_feature}")

    windows = XNPZWindows(args.test_npz, samples_per_step)
    expected_antennas = config.get("num_antennas")
    if expected_antennas is not None and int(expected_antennas) != windows.num_antennas:
        raise ValueError(
            f"Checkpoint was trained with {expected_antennas} antennas, but test X has "
            f"{windows.num_antennas}. Use the matching checkpoint/dataset."
        )
    dataset = PerAntennaSequenceDataset(windows, gain_scale)
    origins = select_origins(
        dataset.sequence_length, pred_steps, args.num_origins
    )
    print(
        f"[TEST DATA] {args.test_npz.resolve()} | X={windows.X.shape} | "
        f"scenarios={windows.scenario_ids}"
    )
    print(
        f"[SEQUENCES] windows={len(windows)}, antennas={windows.num_antennas}, "
        f"per-antenna sequences={len(dataset)}, S={dataset.sequence_length}"
    )
    print(
        f"[CONFIG] samples_per_step={samples_per_step}, "
        f"step={windows.dt * samples_per_step * 1000.0:.3f}ms, "
        f"K={pred_steps}, horizon={windows.dt * samples_per_step * pred_steps * 1000.0:.3f}ms, "
        f"origins={len(origins)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    model = build_model(checkpoint, token_dim).to(device)
    model.eval()
    results, evm_db, reference_gain = evaluate_all_methods(
        model=model,
        loader=loader,
        device=device,
        origins=origins,
        taps=taps,
        gain_scale=gain_scale,
        eps_norm=eps_norm,
        pred_steps=pred_steps,
        loss_gamma=loss_gamma,
        max_age_feature=max_age_feature,
        kf_r_relative=args.kf_r_relative,
        kf_q_relative=args.kf_q_relative,
    )
    for method in ("MODEL", "LR", "KF"):
        print_method(method, results[method], pred_steps)

    args.evm_samples_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.evm_samples_output,
        model_evm_db=evm_db["MODEL"],
        linear_evm_db=evm_db["LR"],
        kalman_evm_db=evm_db["KF"],
        reference_rms_gain=reference_gain,
    )
    print(f"[EVM SAMPLES] {args.evm_samples_output.resolve()}")

    example: Optional[Dict[str, torch.Tensor]] = None
    if not args.no_micro or not args.no_plot:
        if not 0 <= args.example_index < len(dataset):
            raise ValueError(
                f"example-index must be 0..{len(dataset) - 1}, "
                f"got {args.example_index}"
            )
        window_index, antenna_index = dataset.item_location(args.example_index)
        print(
            f"\n[EXAMPLE] dataset_index={args.example_index}, "
            f"window={window_index}, antenna={antenna_index}"
        )
        example = make_micro_example(
            model=model,
            sequence=dataset[args.example_index],
            t0=args.example_t0,
            device=device,
            taps=taps,
            gain_scale=gain_scale,
            eps_norm=eps_norm,
            pred_steps=pred_steps,
            max_age_feature=max_age_feature,
            kf_r_relative=args.kf_r_relative,
            kf_q_relative=args.kf_q_relative,
        )
    if not args.no_micro and example is not None:
        print_micro_example(example, sample_ms=windows.dt * 1000.0)

    plot_metadata: Optional[Dict[str, float]] = None
    if not args.no_plot and example is not None:
        if args.cdf_min_db >= args.cdf_max_db:
            raise ValueError("cdf-min-db must be less than cdf-max-db")
        plot_metadata = plot_prediction_and_evm_cdf(
            micro=example,
            evm_db=evm_db,
            reference_gain=reference_gain,
            sample_ms=windows.dt * 1000.0,
            output_path=args.plot_output,
            history_ms=args.plot_history_ms,
            tail_ms=args.plot_tail_ms,
            cdf_min_db=args.cdf_min_db,
            cdf_max_db=args.cdf_max_db,
        )
        print(f"[PLOT] {args.plot_output.resolve()}")

    evm_summary = {
        method: {
            "count": int(len(values)),
            "mean_db": float(np.mean(values)),
            "median_db": float(np.median(values)),
            "p90_db": float(np.quantile(values, 0.90)),
            "p95_db": float(np.quantile(values, 0.95)),
        }
        for method, values in evm_db.items()
    }

    output = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "test_npz": str(args.test_npz.resolve()),
        "test_shape": list(windows.X.shape),
        "scenario_ids": windows.scenario_ids,
        "num_antennas": windows.num_antennas,
        "samples_per_step": samples_per_step,
        "pred_steps": pred_steps,
        "loss_gamma": loss_gamma,
        "num_origins": len(origins),
        "origins": origins,
        "metrics": results,
        "evm_definition": "20*log10(sqrt(sum(|prediction-GT|^2)/sum(|GT|^2))) per sequence and origin",
        "evm_summary_db": evm_summary,
        "evm_samples_npz": str(args.evm_samples_output.resolve()),
        "plot_png": None if args.no_plot else str(args.plot_output.resolve()),
        "plot_metadata": plot_metadata,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[RESULTS] {args.output_json.resolve()}")
    return results


if __name__ == "__main__":
    main()
