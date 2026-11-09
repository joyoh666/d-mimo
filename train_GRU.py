#!/usr/bin/env python3
"""
Train Simple GRU/RNN channel predictor (per-Rx) in (gain, cos, sin) domain
for FIXED K-step prediction with an age counter:

Given ground-truth tokens up to some step t0, we do:

  Step-1 (teacher input):  x0 = GT[t0]          (age=0) -> predict y1 ~= GT[t0+1]
  Step-2 (autoregressive): x1 = pred(y1)        (age=1) -> predict y2 ~= GT[t0+2]
  Step-3 (autoregressive): x2 = pred(y2)        (age=2) -> predict y3 ~= GT[t0+3]
  ...
  Step-K (autoregressive): x_{K-1} = pred(y_{K-1}) (age=K-1) -> predict yK ~= GT[t0+K]

Age feature:
  age = # of consecutive AR-fed inputs since last GT input
  age_norm = clip(age, MAX_AGE_FEATURE)/MAX_AGE_FEATURE in [0,1]
  RNN input = concat(token (TOK_DIM), age_norm (1)) -> (TOK_DIM+1,)
  RNN output = next token (TOK_DIM,) only.

Loss:
  weights decay rapidly by 0.5 each step:
    w1=1.0, w2=0.5, w3=0.25, ...
  Loss = (sum_k w_k * MSE(y_k, GT[t0+k])) / (sum_k w_k)
  (same for every batch; no scheduled sampling other than pure AR after step1)

Token per step (taps=SAMPLES_PER_STEP, each tap=5ms):
  token_t = [ gain*GAIN_SCALE (taps), cos (taps), sin (taps) ]  -> (3*taps,)

Run:
  python train_simple_rnn_gcs_kstep_age_decay.py
"""

from __future__ import annotations
import os
import time
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# USER CONFIG
# =========================
gpu_id = 0
DEVICE = torch.device(
    f"cuda:{gpu_id}"
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count()
    else "cpu"
)

# Prebuilt datasets only. Dataset generation is intentionally not included here.
DATASET_ROOT = Path("./data_ready/dmimo_3tx_s66_s75/predictor")
TRAIN_X_NPZ = DATASET_ROOT / "train" / "X.npz"
VALIDATION_X_NPZ = DATASET_ROOT / "validation" / "X.npz"
TEST_X_NPZ = DATASET_ROOT / "test" / "X.npz"

SEED = 123

BATCH_SIZE = 256
EPOCHS = 1000
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# RNN config
RNN_TYPE = "GRU"          # "GRU" or "RNN"
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0             # between layers if NUM_LAYERS > 1

# Step definition
SAMPLES_PER_STEP = 5      # 5 * 5ms = 25ms
TAPS = SAMPLES_PER_STEP
GAIN_DIM = TAPS
PHASE_DIM = TAPS
TOK_DIM = 3 * TAPS        # e.g., taps=8 -> TOK_DIM=24

# Token scaling + stability
GAIN_SCALE = 20.0
EPS_NORM = 1e-6

# Train augmentation
AUGMENT_RANDOM_GLOBAL_PHASE = True

# AR feedback stability
DETACH_PRED_INPUT = True

# ===== K-step + age + fast-decaying loss weights =====
PRED_STEPS = 5            # predict 5 steps ahead (K)
LOSS_GAMMA = 0.5          # per-step weight multiplier (0.5 => halves each step)
USE_AGE_FEATURE = True
MAX_AGE_FEATURE = 32
# =====================================================

# Logging / eval / saving
LOG_TRAIN_EVERY_STEPS = 50
EVAL_EVERY_STEPS = 500
EVAL_BATCHES = 0  # 0=all test batches, else limit

SAVE_DIR = "./checkpoints_rnn"
SAVE_EVERY_EPOCHS = 20
# =========================


# -------------------------
# Token helpers
# -------------------------
def complex_to_gcs_tokens(h_taps: torch.Tensor) -> torch.Tensor:
    """
    h_taps: (...,TAPS) complex
    returns (...,3*TAPS) float32 = [|h|*GAIN_SCALE, cos(angle), sin(angle)]
    """
    mag = h_taps.abs() * float(GAIN_SCALE)
    ang = torch.angle(h_taps)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    return torch.cat([mag, cos, sin], dim=-1).to(torch.float32)


def postprocess_pred_tokens(tok: torch.Tensor) -> torch.Tensor:
    """
    tok: (...,3*TAPS) raw
      gain -> softplus
      cos/sin -> normalize per tap
    """
    gain_raw = tok[..., :GAIN_DIM]
    cos_raw  = tok[..., GAIN_DIM:GAIN_DIM + PHASE_DIM]
    sin_raw  = tok[..., GAIN_DIM + PHASE_DIM:GAIN_DIM + 2 * PHASE_DIM]

    gain = F.softplus(gain_raw)

    denom = torch.sqrt(cos_raw * cos_raw + sin_raw * sin_raw + float(EPS_NORM))
    cos_n = cos_raw / denom
    sin_n = sin_raw / denom

    return torch.cat([gain, cos_n, sin_n], dim=-1)


def tokens_to_complex(tok: torch.Tensor) -> torch.Tensor:
    """tok: (...,3*TAPS) in [gain_scaled, cos, sin] -> (...,TAPS) complex taps (gain unscaled)."""
    gain_s = tok[..., :GAIN_DIM]
    cos    = tok[..., GAIN_DIM:GAIN_DIM + PHASE_DIM]
    sin    = tok[..., GAIN_DIM + PHASE_DIM:GAIN_DIM + 2 * PHASE_DIM]
    gain = gain_s / float(GAIN_SCALE)
    return torch.complex(gain * cos, gain * sin)


# -------------------------
# Dataset
# -------------------------
class XNPZWindows(Dataset):
    """Read prebuilt windows with shape (N, L, C), where C is inferred."""

    def __init__(self, x_npz_path: str | Path):
        self.path = str(x_npz_path)
        if not Path(self.path).is_file():
            raise FileNotFoundError(self.path)
        self._npz = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if "X" not in self._npz:
            raise KeyError(f"'X' not found in {self.path}")

        self.X = self._npz["X"]  # (N,L,C) complex64
        if self.X.ndim != 3 or self.X.shape[-1] <= 0:
            raise ValueError(
                f"Expected X shape (N,L,C) with C > 0, got {self.X.shape}"
            )
        if not np.issubdtype(self.X.dtype, np.complexfloating):
            raise TypeError(f"Expected complex CSI, got dtype={self.X.dtype}")

        self.num_antennas = int(self.X.shape[-1])
        if "num_antennas" in self._npz:
            metadata_count = int(self._npz["num_antennas"])
            if metadata_count != self.num_antennas:
                raise ValueError(
                    f"num_antennas metadata={metadata_count}, X has "
                    f"{self.num_antennas} channels"
                )
        self.dt = float(self._npz["dt"]) if "dt" in self._npz else 0.005

        L = int(self.X.shape[1])
        if L % SAMPLES_PER_STEP != 0:
            raise ValueError(
                f"L={L} not divisible by SAMPLES_PER_STEP={SAMPLES_PER_STEP}"
            )

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = np.array(self.X[idx], copy=False)
        return torch.from_numpy(x)  # (L,C) complex64 CPU


class PerAntennaSequenceDataset(Dataset):
    """Return one token sequence for each (window, antenna) pair."""

    def __init__(
        self,
        windows_ds: XNPZWindows,
        window_indices: np.ndarray,
        train: bool,
    ):
        self.windows_ds = windows_ds
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.train = bool(train)
        self.num_antennas = windows_ds.num_antennas
        if len(self.window_indices) == 0:
            raise ValueError("window_indices must not be empty")

        sample0 = self.windows_ds[int(self.window_indices[0])]
        L = int(sample0.shape[0])
        self.S = L // SAMPLES_PER_STEP

    def __len__(self) -> int:
        return int(len(self.window_indices) * self.num_antennas)

    def __getitem__(self, idx: int) -> torch.Tensor:
        window_local = idx // self.num_antennas
        antenna_idx = idx % self.num_antennas
        window_idx = int(self.window_indices[window_local])

        H = self.windows_ds[window_idx]  # (L,C) complex
        h = H[:, antenna_idx]            # (L,) complex

        if self.train and AUGMENT_RANDOM_GLOBAL_PHASE:
            phi = torch.rand((), dtype=torch.float32) * (2.0 * math.pi)
            phasor = torch.polar(torch.ones((), dtype=torch.float32), phi)
            h = h * phasor.to(h.dtype)

        h_taps = h.reshape(self.S, SAMPLES_PER_STEP)
        return complex_to_gcs_tokens(h_taps)


# -------------------------
# Model: GRU/RNN + projection (token + age_norm -> token)
# -------------------------
class SimpleChannelRNN(nn.Module):
    def __init__(self, tok_dim: int, hidden: int, layers: int, dropout: float, rnn_type: str = "GRU",
                 use_age_feature: bool = True):
        super().__init__()
        self.tok_dim = int(tok_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.rnn_type = rnn_type.upper()
        self.use_age_feature = bool(use_age_feature)

        self.in_dim = self.tok_dim + (1 if self.use_age_feature else 0)

        self.init_tok = nn.Parameter(torch.zeros(self.tok_dim))
        nn.init.normal_(self.init_tok, std=0.02)

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.in_dim,
                hidden_size=self.hidden,
                num_layers=self.layers,
                batch_first=True,
                dropout=(dropout if self.layers > 1 else 0.0),
            )
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=self.in_dim,
                hidden_size=self.hidden,
                num_layers=self.layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=(dropout if self.layers > 1 else 0.0),
            )
        else:
            raise ValueError(f"Unknown RNN_TYPE={rnn_type}")

        self.out_proj = nn.Linear(self.hidden, self.tok_dim)

    def _pack_in(self, tok: torch.Tensor, age_norm: torch.Tensor | None) -> torch.Tensor:
        if not self.use_age_feature:
            return tok
        assert age_norm is not None
        return torch.cat([tok, age_norm], dim=-1)

    def rnn_step(self, tok_in: torch.Tensor, age_norm: torch.Tensor | None, h: torch.Tensor | None):
        """
        One-step transition.
          tok_in: (B,TOK_DIM)
          age_norm: (B,1) or None
          h: (layers,B,H) or None
        Returns:
          out: (B,H), h_next
        """
        x = self._pack_in(tok_in, age_norm).unsqueeze(1)  # (B,1,in_dim)
        y, h_next = self.rnn(x, h)                         # y: (B,1,H)
        return y[:, 0, :], h_next


# -------------------------
# K-step training loss (pure AR after the first GT input) + age counter
# -------------------------
def k_step_loss(model: SimpleChannelRNN, seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    seq: (B,S,TOK_DIM) GT tokens
    Choose random t0 in [0, S-(PRED_STEPS+1)].

    Warmup hidden with init + GT[0..t0-1] (age=0).
    Then:
      step1 input = GT[t0], age=0
      stepk input = pred_{k-1}, age=k-1 for k>=2

    Weighted loss:
      weights: w_k = LOSS_GAMMA^(k-1)  (k=1..K)
      loss = sum(w_k * MSE_k) / sum(w_k)

    Returns (loss, debug_stats)
    """
    B, S, D = seq.shape
    assert D == TOK_DIM

    K = int(PRED_STEPS)
    if S < (K + 1):
        return torch.zeros((), device=seq.device), {"t0": -1, "sum_w": 0.0}

    # randint high is exclusive; want t0 <= S-K-1 => high = S-K
    t0 = int(torch.randint(low=0, high=(S - K), size=(1,), device=seq.device).item())

    # prefix to set hidden after consuming GT[0..t0-1], age=0
    init_tok = model.init_tok.view(1, 1, D).expand(B, 1, D)  # (B,1,D)
    if model.use_age_feature:
        age0_init = torch.zeros((B, 1, 1), device=seq.device, dtype=seq.dtype)
        init_in = torch.cat([init_tok, age0_init], dim=-1)   # (B,1,D+1)
    else:
        init_in = init_tok

    if t0 > 0:
        prefix_tok = seq[:, :t0, :]  # (B,t0,D)
        if model.use_age_feature:
            prefix_age0 = torch.zeros((B, t0, 1), device=seq.device, dtype=seq.dtype)
            prefix_in = torch.cat([prefix_tok, prefix_age0], dim=-1)  # (B,t0,D+1)
        else:
            prefix_in = prefix_tok
        prefix_full = torch.cat([init_in, prefix_in], dim=1)
    else:
        prefix_full = init_in

    _, h = model.rnn(prefix_full)  # h is state BEFORE consuming GT[t0]

    # weights
    w = torch.tensor([float(LOSS_GAMMA) ** i for i in range(K)], device=seq.device, dtype=seq.dtype)  # (K,)
    sum_w = float(w.sum().item())

    # rollout
    loss_acc = torch.zeros((), device=seq.device, dtype=seq.dtype)

    x = seq[:, t0, :]  # step1 input = GT[t0]
    for i in range(K):
        age_val = i  # 0,1,2,3,4...
        if model.use_age_feature:
            age_norm = (torch.full((B,), age_val, device=seq.device, dtype=torch.long)
                        .clamp(0, MAX_AGE_FEATURE).to(seq.dtype) / float(MAX_AGE_FEATURE)).unsqueeze(-1)  # (B,1)
        else:
            age_norm = None

        out, h = model.rnn_step(x, age_norm, h)
        pred = postprocess_pred_tokens(model.out_proj(out))  # (B,D)
        tgt = seq[:, t0 + i + 1, :]                          # GT[t0+i+1]

        mse_i = F.mse_loss(pred, tgt, reduction="mean")
        loss_acc = loss_acc + w[i] * mse_i

        # next input (AR)
        if i != (K - 1):
            x = pred.detach() if DETACH_PRED_INPUT else pred

    loss = loss_acc / max(sum_w, 1e-12)
    return loss, {"t0": float(t0), "sum_w": sum_w}


@torch.no_grad()
def eval_loader_kstep(model: SimpleChannelRNN, loader: DataLoader, device: torch.device, eval_batches: int = 0) -> Dict[str, float]:
    """
    Random t0 per batch, same K-step procedure as training.
    Reports weighted averages over steps:
      - tok_mse (token domain)
      - complex_mse
      - nmse
    """
    was_training = model.training
    model.eval()

    tok_sum = 0.0
    cmse_sum = 0.0
    nmse_sum = 0.0
    n = 0

    K = int(PRED_STEPS)
    w = torch.tensor([float(LOSS_GAMMA) ** i for i in range(K)], device=device, dtype=torch.float32)
    w_sum = float(w.sum().item())

    for bi, seq in enumerate(loader):
        if eval_batches > 0 and bi >= eval_batches:
            break

        seq = seq.to(device, non_blocking=True)  # (B,S,D)
        B, S, D = seq.shape
        if S < (K + 1):
            continue

        t0 = int(torch.randint(low=0, high=(S - K), size=(1,), device=device).item())

        # prefix (age=0)
        init_tok = model.init_tok.view(1, 1, D).expand(B, 1, D)
        if model.use_age_feature:
            init_in = torch.cat([init_tok, torch.zeros((B, 1, 1), device=device, dtype=seq.dtype)], dim=-1)
        else:
            init_in = init_tok

        if t0 > 0:
            prefix_tok = seq[:, :t0, :]
            if model.use_age_feature:
                prefix_in = torch.cat([prefix_tok, torch.zeros((B, t0, 1), device=device, dtype=seq.dtype)], dim=-1)
            else:
                prefix_in = prefix_tok
            prefix_full = torch.cat([init_in, prefix_in], dim=1)
        else:
            prefix_full = init_in

        _, h = model.rnn(prefix_full)

        # rollout and accumulate weighted metrics
        x = seq[:, t0, :]
        tok_acc = 0.0
        cmse_acc = 0.0
        denom_acc = 0.0
        for i in range(K):
            age_val = i
            if model.use_age_feature:
                age_norm = (torch.full((B,), age_val, device=device, dtype=torch.long)
                            .clamp(0, MAX_AGE_FEATURE).to(seq.dtype) / float(MAX_AGE_FEATURE)).unsqueeze(-1)
            else:
                age_norm = None

            out, h = model.rnn_step(x, age_norm, h)
            pred = postprocess_pred_tokens(model.out_proj(out))
            tgt = seq[:, t0 + i + 1, :]

            wi = float(w[i].item())

            tok_acc += wi * F.mse_loss(pred, tgt, reduction="mean").item()

            pred_h = tokens_to_complex(pred)
            tgt_h  = tokens_to_complex(tgt)
            cmse_acc += wi * ((pred_h - tgt_h).abs() ** 2).mean().item()
            denom_acc += wi * (tgt_h.abs() ** 2).mean().item()
            if i != (K - 1):
                x = pred.detach() if DETACH_PRED_INPUT else pred

        tok_mse = tok_acc / max(w_sum, 1e-12)
        cmse = cmse_acc / max(w_sum, 1e-12)
        nmse = cmse / max(denom_acc / max(w_sum, 1e-12), 1e-12)
        tok_sum += tok_mse
        cmse_sum += cmse
        nmse_sum += nmse
        n += 1

    if was_training:
        model.train()

    if n == 0:
        return {"tok_mse": float("nan"), "complex_mse": float("nan"), "nmse": float("nan")}
    return {"tok_mse": tok_sum / n, "complex_mse": cmse_sum / n, "nmse": nmse_sum / n}


# -------------------------
# Main
# -------------------------
def _checkpoint_payload(
    model: SimpleChannelRNN,
    optim: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    num_antennas: int,
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "train_x_path": str(TRAIN_X_NPZ),
        "validation_x_path": str(VALIDATION_X_NPZ),
        "test_x_path": str(TEST_X_NPZ),
        "config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "rnn_type": RNN_TYPE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "samples_per_step": SAMPLES_PER_STEP,
            "tok_dim": TOK_DIM,
            "gain_scale": GAIN_SCALE,
            "eps_norm": EPS_NORM,
            "augment_random_global_phase": AUGMENT_RANDOM_GLOBAL_PHASE,
            "detach_pred_input": DETACH_PRED_INPUT,
            "pred_steps": PRED_STEPS,
            "loss_gamma": LOSS_GAMMA,
            "use_age_feature": USE_AGE_FEATURE,
            "max_age_feature": MAX_AGE_FEATURE,
            "num_antennas": num_antennas,
        },
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[DEVICE] {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[CUDA] name: {torch.cuda.get_device_name(DEVICE)}")

    train_windows = XNPZWindows(TRAIN_X_NPZ)
    validation_windows = XNPZWindows(VALIDATION_X_NPZ)
    test_windows = XNPZWindows(TEST_X_NPZ)

    expected_shape = train_windows.X.shape[1:]
    for split_name, windows in (
        ("validation", validation_windows),
        ("test", test_windows),
    ):
        if windows.X.shape[1:] != expected_shape:
            raise ValueError(
                f"{split_name} window shape {windows.X.shape[1:]} does not "
                f"match train shape {expected_shape}"
            )
        if not math.isclose(windows.dt, train_windows.dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{split_name} dt={windows.dt} does not match train dt={train_windows.dt}"
            )

    train_indices = np.arange(len(train_windows), dtype=np.int64)
    validation_indices = np.arange(len(validation_windows), dtype=np.int64)
    test_indices = np.arange(len(test_windows), dtype=np.int64)

    train_ds = PerAntennaSequenceDataset(train_windows, train_indices, train=True)
    validation_ds = PerAntennaSequenceDataset(
        validation_windows, validation_indices, train=False
    )
    test_ds = PerAntennaSequenceDataset(test_windows, test_indices, train=False)

    print(
        f"[DATA] antennas={train_windows.num_antennas}, "
        f"window_shape={expected_shape}, dt={train_windows.dt:.6f}s"
    )
    print(
        f"[SPLIT] train={len(train_windows)} windows/{len(train_ds)} sequences, "
        f"validation={len(validation_windows)} windows/{len(validation_ds)} sequences, "
        f"test={len(test_windows)} windows/{len(test_ds)} sequences"
    )

    L = int(train_windows[0].shape[0])
    S = train_ds.S
    step_ms = train_windows.dt * 1000.0 * SAMPLES_PER_STEP
    print(
        f"[SEQ] L={L}, samples_per_step={SAMPLES_PER_STEP}, "
        f"step={step_ms:.3f}ms, S={S}, TOK_DIM={TOK_DIM}"
    )
    print(
        f"[KSTEP] K={PRED_STEPS}, LOSS_GAMMA={LOSS_GAMMA}, "
        f"age_feature={USE_AGE_FEATURE}"
    )
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
        drop_last=(len(train_ds) >= BATCH_SIZE),
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )

    model = SimpleChannelRNN(
        tok_dim=TOK_DIM,
        hidden=HIDDEN_SIZE,
        layers=NUM_LAYERS,
        dropout=DROPOUT,
        rnn_type=RNN_TYPE,
        use_age_feature=USE_AGE_FEATURE,
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path = os.path.join(SAVE_DIR, "best_validation_nmse.pt")
    best_val_nmse = float("inf")
    global_step = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        run_n = 0

        for seq in train_loader:
            seq = seq.to(DEVICE, non_blocking=True)
            loss, dbg = k_step_loss(model, seq)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

            global_step += 1
            run_loss += float(loss.item())
            run_n += 1

            if LOG_TRAIN_EVERY_STEPS > 0 and global_step % LOG_TRAIN_EVERY_STEPS == 0:
                elapsed = time.time() - t_start
                print(
                    f"[TRAIN step {global_step:7d} | epoch {epoch:4d}] "
                    f"lossK={run_loss / max(run_n, 1):.6e} "
                    f"(t0={int(dbg['t0'])}) | t={elapsed / 60:.1f} min"
                )
                run_loss = 0.0
                run_n = 0

            if EVAL_EVERY_STEPS > 0 and global_step % EVAL_EVERY_STEPS == 0:
                metrics = eval_loader_kstep(
                    model, validation_loader, DEVICE, eval_batches=EVAL_BATCHES
                )
                print(
                    f"[VAL step {global_step:7d}] tok_mse={metrics['tok_mse']:.3e} "
                    f"cmse={metrics['complex_mse']:.3e} nmse={metrics['nmse']:.3e}"
                )

        val_metrics = eval_loader_kstep(
            model, validation_loader, DEVICE, eval_batches=EVAL_BATCHES
        )
        message = (
            f"[EPOCH {epoch:4d}] validation tok_mse={val_metrics['tok_mse']:.3e} "
            f"cmse={val_metrics['complex_mse']:.3e} nmse={val_metrics['nmse']:.3e}"
        )

        if math.isfinite(val_metrics["nmse"]) and val_metrics["nmse"] < best_val_nmse:
            best_val_nmse = val_metrics["nmse"]
            torch.save(
                _checkpoint_payload(
                    model,
                    optim,
                    epoch,
                    global_step,
                    train_windows.num_antennas,
                ),
                best_path,
            )
            message += f" | best saved: {best_path}"

        if SAVE_EVERY_EPOCHS > 0 and epoch % SAVE_EVERY_EPOCHS == 0:
            periodic_path = os.path.join(
                SAVE_DIR,
                f"rnn_gcs_kstep_age_gamma{LOSS_GAMMA:.2f}_K{PRED_STEPS}_"
                f"epoch{epoch:04d}.pt",
            )
            torch.save(
                _checkpoint_payload(
                    model,
                    optim,
                    epoch,
                    global_step,
                    train_windows.num_antennas,
                ),
                periodic_path,
            )
            message += f" | periodic saved: {periodic_path}"

        print(message)

    if not os.path.isfile(best_path):
        raise RuntimeError("No best checkpoint was saved")
    best_checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = eval_loader_kstep(model, test_loader, DEVICE, eval_batches=0)
    print(
        f"[FINAL TEST] best_epoch={best_checkpoint['epoch']} "
        f"tok_mse={test_metrics['tok_mse']:.3e} "
        f"cmse={test_metrics['complex_mse']:.3e} "
        f"nmse={test_metrics['nmse']:.3e}"
    )


if __name__ == "__main__":
    main()
