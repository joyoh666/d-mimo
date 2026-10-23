# [YT 1019] 형주형 Final_Train_260217_RNN_fivestep.ipynb 첫 셀을 그대로 옮긴 것. 바꾼 줄에만 [YT 1019] 표시.
#           cd live_demo 후 python train_predictor.py
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
import glob
import time
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# USER CONFIG
# =========================
gpu_id = 0  # [YT 1019] 이 PC GPU 하나
DEVICE = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

DATA_READY_ROOT = "./data_ready/pred_ws6p00s"  # [YT 1019] build_dataset.py 출력
X_NPZ_EXPLICIT = ""  # optional: exact X.npz path; else newest under DATA_READY_ROOT

SEED = 123

BATCH_SIZE = 256
EPOCHS = int(os.environ.get("EPOCHS", "1000"))  # [YT 1019] 환경변수로 줄일 수 있게
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
# Utils
# -------------------------
def find_x_npz(data_ready_root: str) -> str:
    cands = glob.glob(os.path.join(data_ready_root, "**", "X.npz"), recursive=True)
    if not cands:
        raise FileNotFoundError(f"No X.npz found under: {os.path.abspath(data_ready_root)}")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


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
    def __init__(self, x_npz_path: str):
        self.path = x_npz_path
        self._npz = np.load(self.path, mmap_mode="r")
        if "X" not in self._npz:
            raise KeyError(f"'X' not found in {self.path}")
        self.X = self._npz["X"]  # (N,L,num_ants) complex64
        if self.X.ndim != 3 or self.X.shape[-1] < 1:
            raise ValueError(f"Expected X shape (N,L,num_ants), got {self.X.shape}")
        self.num_ants = int(self.X.shape[-1])

        L = self.X.shape[1]
        if (L % SAMPLES_PER_STEP) != 0:
            raise ValueError(f"L={L} not divisible by SAMPLES_PER_STEP={SAMPLES_PER_STEP}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = np.array(self.X[idx], copy=False)
        return torch.from_numpy(x)  # (L,num_ants) complex64 CPU


class PerRxSequenceDataset(Dataset):
    """Returns seq: (S,TOK_DIM) float32 tokens for a (window, rx)."""
    def __init__(self, windows_ds: XNPZWindows, window_indices: np.ndarray, train: bool):
        self.windows_ds = windows_ds
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.train = bool(train)
        self.num_ants = windows_ds.num_ants

        sample0 = self.windows_ds[int(self.window_indices[0])]
        L = int(sample0.shape[0])
        self.S = L // SAMPLES_PER_STEP

    def __len__(self) -> int:
        return int(len(self.window_indices) * self.num_ants)

    def __getitem__(self, idx: int) -> torch.Tensor:
        w_local = idx // self.num_ants
        rx = idx % self.num_ants
        w_idx = int(self.window_indices[w_local])

        H = self.windows_ds[w_idx]  # (L,num_ants) complex
        h = H[:, rx]                # (L,) complex

        if self.train and AUGMENT_RANDOM_GLOBAL_PHASE:
            phi = torch.rand((), dtype=torch.float32) * (2.0 * math.pi)
            phasor = torch.polar(torch.ones((), dtype=torch.float32), phi)  # unit phasor
            h = h * phasor.to(h.dtype)

        h_taps = h.reshape(self.S, SAMPLES_PER_STEP)  # (S,TAPS) complex
        seq = complex_to_gcs_tokens(h_taps)           # (S,3*TAPS) float32
        return seq


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
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[DEVICE] {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[CUDA] name: {torch.cuda.get_device_name(DEVICE)}")

    x_path = X_NPZ_EXPLICIT.strip() if X_NPZ_EXPLICIT.strip() else find_x_npz(DATA_READY_ROOT)
    print(f"[DATA] Using: {x_path}")

    windows = XNPZWindows(x_path)
    N = len(windows)

    manifest_path = os.path.join(os.path.dirname(x_path), "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing episode split manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path).sort_values("sample_id").reset_index(drop=True)
    if len(manifest) != N or "split" not in manifest.columns:
        raise ValueError("manifest.csv must match X and contain a 'split' column")
    split_values = manifest["split"].astype(str).to_numpy()
    train_indices = np.flatnonzero(split_values == "train")
    validation_indices = np.flatnonzero(split_values == "validation")
    test_indices = np.flatnonzero(split_values == "test")
    if min(len(train_indices), len(validation_indices), len(test_indices)) == 0:
        raise ValueError("train, validation, and test splits must all be non-empty")

    print(
        f"[SPLIT] windows N={N} => train={len(train_indices)} "
        f"validation={len(validation_indices)} test={len(test_indices)} "
        f"(expanded x{windows.num_ants} per-Rx)"
    )

    train_ds = PerRxSequenceDataset(windows, train_indices, train=True)
    validation_ds = PerRxSequenceDataset(windows, validation_indices, train=False)
    test_ds  = PerRxSequenceDataset(windows, test_indices,  train=False)

    L = int(windows[0].shape[0])
    S = train_ds.S
    step_ms = 5.0 * SAMPLES_PER_STEP
    print(f"[SEQ] L={L} samples, step={SAMPLES_PER_STEP} -> {step_ms:.1f}ms, S={S} steps, TOK_DIM={TOK_DIM}")
    print(f"[TOK] [gain*{GAIN_SCALE:.1f}, cos, sin] taps={TAPS} | pred: softplus(gain) + normalize(cos,sin)")
    print(f"[KSTEP] K={PRED_STEPS}, LOSS_GAMMA={LOSS_GAMMA} (w: 1,0.5,0.25,...)")
    print(f"[AGE ] enabled={USE_AGE_FEATURE}, MAX_AGE_FEATURE={MAX_AGE_FEATURE} | age=0 on GT input, then 1..K-1 on AR inputs")
    print(f"[AR  ] DETACH_PRED_INPUT={DETACH_PRED_INPUT}")

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin, drop_last=True)
    validation_loader = DataLoader(validation_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin, drop_last=False)

    model = SimpleChannelRNN(
        tok_dim=TOK_DIM,
        hidden=HIDDEN_SIZE,
        layers=NUM_LAYERS,
        dropout=DROPOUT,
        rnn_type=RNN_TYPE,
        use_age_feature=USE_AGE_FEATURE
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    os.makedirs(SAVE_DIR, exist_ok=True)

    global_step = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        run_n = 0

        for seq in train_loader:
            seq = seq.to(DEVICE, non_blocking=True)  # (B,S,TOK_DIM)

            loss, dbg = k_step_loss(model, seq)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

            global_step += 1
            run_loss += float(loss.item())
            run_n += 1

            if LOG_TRAIN_EVERY_STEPS > 0 and (global_step % LOG_TRAIN_EVERY_STEPS == 0):
                elapsed = time.time() - t_start
                print(
                    f"[train step {global_step:7d} | epoch {epoch:4d}] "
                    f"lossK={run_loss/run_n:.6e} (t0={int(dbg['t0'])}) | t={elapsed/60:.1f} min"
                )
                run_loss = 0.0
                run_n = 0

            if EVAL_EVERY_STEPS > 0 and (global_step % EVAL_EVERY_STEPS == 0):
                m = eval_loader_kstep(model, validation_loader, DEVICE, eval_batches=EVAL_BATCHES)
                elapsed = time.time() - t_start
                print(
                    f"[VALIDATION step {global_step:7d} | epoch {epoch:4d}] "
                    f"Kstep tok_mse={m['tok_mse']:.3e} cmse={m['complex_mse']:.3e} nmse={m['nmse']:.3e} "
                    f"| t={elapsed/60:.1f} min"
                )

        # end-of-epoch eval
        m = eval_loader_kstep(model, validation_loader, DEVICE, eval_batches=EVAL_BATCHES)
        msg = (f"[EPOCH {epoch:4d} VALIDATION] Kstep tok_mse={m['tok_mse']:.3e} "
               f"cmse={m['complex_mse']:.3e} nmse={m['nmse']:.3e}")

        # save
        saved_path = None
        if (SAVE_EVERY_EPOCHS is not None) and (SAVE_EVERY_EPOCHS > 0) and (epoch % SAVE_EVERY_EPOCHS == 0):
            saved_path = os.path.join(SAVE_DIR, f"rnn_gcs_kstep_age_gamma{LOSS_GAMMA:.2f}_K{PRED_STEPS}_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "x_path": x_path,
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
                        "data_ready_root": DATA_READY_ROOT,
                        "num_ants": windows.num_ants,
                    }
                }, saved_path)

        print(msg + (f" | saved: {saved_path}" if saved_path else " | (no checkpoint)"))

    test_metrics = eval_loader_kstep(model, test_loader, DEVICE, eval_batches=0)
    print(
        f"[FINAL TEST] Kstep tok_mse={test_metrics['tok_mse']:.3e} "
        f"cmse={test_metrics['complex_mse']:.3e} nmse={test_metrics['nmse']:.3e}"
    )


if __name__ == "__main__":
    main()
