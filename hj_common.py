# -*- coding: utf-8 -*-
"""
형주형 코드(Final_Train/Test_260217_RNN_fivestep, Final_Train/Test_260316_capacity)의 추론 부분을 numpy 로 옮긴 것.
실시간 데모 노트북에 torch 가 없어도 돌도록 했다. 수식·순서는 원본 그대로이고, export_numpy.py 가 torch 출력과 같은지 검사한다.

데이터 모양·크기 맞추기:
  입력 CSI 의 안테나 차원은 그대로 보존한다.
  크기는 SCALE 하나로 맞춘다 (build_dataset.py 가 정해 data_ready/scale.json 에 남김. 학습·실시간 모두 같은 값).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_READY = HERE / "data_ready"
MODEL_DIR = HERE / "models"


def default_logs() -> Path:
    """1017 측정 logs 위치. 노트북(d_mimo_1017_3iso/live_demo)이면 ../logs, 이 PC 정리본이면 ../../3_실험_*/logs."""
    cands = [HERE / "logs", HERE.parent / "logs"] + sorted(HERE.parent.parent.glob("3_실험_*/logs"))
    for p in cands:
        if any(p.rglob("2026*_tb_usrp_3tx_S*_ISO3")):
            return p
    return cands[0]

DT = 0.005
STATE_NAMES = ["RU1", "RU2", "RU3", "RU12", "RU13", "RU23"]
N_ACTIONS = 6
STATE_RUS = {0: [1], 1: [2], 2: [3], 3: [1, 2], 4: [1, 3], 5: [2, 3]}


# -------------------------
# 데이터 모양 · 크기
# -------------------------
def scale_csi(csi: np.ndarray, scale: float) -> np.ndarray:
    """CSI의 모양을 보존하면서 모든 복소 채널 계수에 ``scale``을 적용한다."""
    c = np.asarray(csi, dtype=np.complex64)
    if c.ndim != 2:
        raise ValueError(f"Expected CSI shape (T, num_ants), got {c.shape}")
    return c * np.float32(scale)


def load_scale() -> float:
    return float(json.loads((DATA_READY / "scale.json").read_text(encoding="utf-8"))["scale"])


# -------------------------
# 토큰 (원본 complex_to_gcs_tokens / postprocess_pred_tokens / tokens_to_complex)
# -------------------------
def complex_to_gcs_tokens(h_taps: np.ndarray, gain_scale: float) -> np.ndarray:
    ang = np.angle(h_taps)
    return np.concatenate([np.abs(h_taps) * gain_scale, np.cos(ang), np.sin(ang)], axis=-1).astype(np.float32)


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def postprocess_pred_tokens(tok: np.ndarray, taps: int, eps_norm: float) -> np.ndarray:
    g = softplus(tok[..., :taps])
    c = tok[..., taps:2 * taps]
    s = tok[..., 2 * taps:3 * taps]
    d = np.sqrt(c * c + s * s + eps_norm)
    return np.concatenate([g, c / d, s / d], axis=-1)


def tokens_to_complex(tok: np.ndarray, taps: int, gain_scale: float) -> np.ndarray:
    g = tok[..., :taps] / gain_scale
    return g * tok[..., taps:2 * taps] + 1j * g * tok[..., 2 * taps:3 * taps]


# -------------------------
# GRU (torch.nn.GRU / GRUCell 과 같은 식: 게이트 순서 r, z, n)
# -------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_cell(x, h, w_ih, w_hh, b_ih, b_hh):
    gi = x @ w_ih.T + b_ih
    gh = h @ w_hh.T + b_hh
    H = h.shape[-1]
    r = _sig(gi[..., :H] + gh[..., :H])
    z = _sig(gi[..., H:2 * H] + gh[..., H:2 * H])
    n = np.tanh(gi[..., 2 * H:] + r * gh[..., 2 * H:])
    return (1.0 - z) * n + z * h


class NumpyPredictor:
    """형주형 SimpleChannelRNN (GRU 1층, 토큰 + age) 의 K-step rollout (model_kstep_predict_batch)."""

    def __init__(self, path=MODEL_DIR / "predictor.npz"):
        z = np.load(path, allow_pickle=False)
        self.cfg = json.loads(str(z["config_json"]))
        if int(self.cfg.get("num_layers", 1)) != 1 or str(self.cfg.get("rnn_type", "GRU")).upper() != "GRU":
            raise ValueError("numpy 판은 GRU 1층만 지원")
        self.w_ih, self.w_hh = z["rnn.weight_ih_l0"], z["rnn.weight_hh_l0"]
        self.b_ih, self.b_hh = z["rnn.bias_ih_l0"], z["rnn.bias_hh_l0"]
        self.w_out, self.b_out = z["out_proj.weight"], z["out_proj.bias"]
        self.init_tok = z["init_tok"]
        self.taps = int(self.cfg["samples_per_step"])
        self.K = int(self.cfg["pred_steps"])
        self.gain_scale = float(self.cfg["gain_scale"])
        self.eps_norm = float(self.cfg["eps_norm"])
        self.max_age = int(self.cfg["max_age_feature"])
        self.use_age = bool(self.cfg["use_age_feature"])
        self.hidden = self.w_hh.shape[1]

    def _in(self, tok, age):
        if not self.use_age:
            return tok
        a = np.full(tok.shape[:-1] + (1,), min(age, self.max_age) / float(self.max_age), np.float32)
        return np.concatenate([tok, a], axis=-1)

    def predict(self, seq_tok: np.ndarray) -> np.ndarray:
        """seq_tok (B,S,D) = GT 토큰 S 스텝 (마지막이 GT[t0]). 반환 (B,K,D) = GT[t0+1..t0+K] 예측 (후처리된 토큰)."""
        B, S, D = seq_tok.shape
        h = np.zeros((B, self.hidden), np.float32)
        h = gru_cell(self._in(np.broadcast_to(self.init_tok, (B, D)), 0), h, self.w_ih, self.w_hh, self.b_ih, self.b_hh)
        for t in range(S - 1):                                   # warmup GT[0..t0-1], age 0
            h = gru_cell(self._in(seq_tok[:, t], 0), h, self.w_ih, self.w_hh, self.b_ih, self.b_hh)
        x = seq_tok[:, -1]
        out = []
        for i in range(self.K):
            h = gru_cell(self._in(x, i), h, self.w_ih, self.w_hh, self.b_ih, self.b_hh)
            p = postprocess_pred_tokens(h @ self.w_out.T + self.b_out, self.taps, self.eps_norm)
            out.append(p)
            x = p
        return np.stack(out, axis=1)

    def predict_complex(self, h_hist: np.ndarray) -> np.ndarray:
        """h_hist (B, S*taps) complex (안테나 한 열) -> (B, K*taps) complex 예측 (같은 크기 단위)."""
        B, L = h_hist.shape
        S = L // self.taps
        taps = np.asarray(h_hist[:, L - S * self.taps:]).reshape(B, S, self.taps)
        tok = complex_to_gcs_tokens(taps, self.gain_scale)
        pred = self.predict(tok)
        return tokens_to_complex(pred, self.taps, self.gain_scale).reshape(B, -1)


class NumpyDQN:
    """형주형 GRUQNetwork (GRUCell + q_head) — Final_Test_260316 의 model_greedy."""

    def __init__(self, path=MODEL_DIR / "dqn.npz"):
        z = np.load(path, allow_pickle=False)
        self.cfg = json.loads(str(z["env_cfg_json"]))
        self.w_ih, self.w_hh = z["gru.weight_ih"], z["gru.weight_hh"]
        self.b_ih, self.b_hh = z["gru.bias_ih"], z["gru.bias_hh"]
        self.w_q, self.b_q = z["q_head.weight"], z["q_head.bias"]
        self.h0 = z["h0"]
        self.reset()

    def reset(self):
        self.h = self.h0.astype(np.float32).copy()[None, :]

    def step(self, obs: np.ndarray):
        self.h = gru_cell(obs[None, :].astype(np.float32), self.h, self.w_ih, self.w_hh, self.b_ih, self.b_hh)
        q = (self.h @ self.w_q.T + self.b_q)[0]
        return int(np.argmax(q)), q


# -------------------------
# 환경 식 (RUWindowEnv 의 ru_gain_segment / all_state_power_segment / _make_obs)
# -------------------------
def build_ru_antenna_map(num_ants: int, num_rus: int = 3) -> dict[int, list[int]]:
    """안테나 열을 고정된 RU들에 연속적이고 가능한 한 균등하게 배분한다."""
    if num_ants < num_rus:
        raise ValueError(f"num_ants={num_ants} must be >= num_rus={num_rus}")
    splits = np.array_split(np.arange(num_ants, dtype=np.int64), num_rus)
    return {ru + 1: split.tolist() for ru, split in enumerate(splits)}


def ru_gain_segment(g_seg: np.ndarray) -> np.ndarray:
    """g_seg (segL,num_ants) |X| -> 3개 RU의 구간 평균 이득."""
    s = np.asarray(g_seg, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"Expected gain segment shape (segL, num_ants), got {s.shape}")
    ru_ants = build_ru_antenna_map(s.shape[1])
    return np.array([
        np.sqrt(np.sum(s[:, ru_ants[ru]] ** 2, axis=1)).mean()
        for ru in range(1, 4)
    ], np.float32)


def all_state_power_segment(g_seg: np.ndarray, phase_std: float, noise_power: float, reward_snr_db: float = 10.0,
                            rng=np.random) -> np.ndarray:
    s = np.asarray(g_seg, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"Expected gain segment shape (segL, num_ants), got {s.shape}")
    ru_ants = build_ru_antenna_map(s.shape[1])
    p = [np.sum(s[:, ru_ants[ru]] ** 2, axis=1) for ru in range(1, 4)]

    def pair(pa, pb):
        a = np.sqrt(np.maximum(pa, 0.0) * 0.5)
        b = np.sqrt(np.maximum(pb, 0.0) * 0.5)
        dphi = rng.normal(0.0, phase_std, size=a.shape)
        return float(np.mean(a * a + b * b + 2.0 * a * b * np.cos(dphi)))

    power = np.array([p[0].mean(), p[1].mean(), p[2].mean(), pair(p[0], p[1]), pair(p[0], p[2]), pair(p[1], p[2])])
    npow = noise_power * (10.0 ** ((10.0 - reward_snr_db) / 10.0))
    return np.log2(1.0 + power / npow).astype(np.float32)


def make_obs(ru_gain_hist, conn_hist, seg_k: int, conn_state: int, hist_bins: int = 3, mask_value: float = 0.0):
    """ru_gain_hist[j] = (3,) 세그먼트 j 의 RU 이득, conn_hist[j] = 세그먼트 j 를 서비스한 state."""
    obs = np.full((3, hist_bins), mask_value, np.float32)
    n = len(ru_gain_hist)
    for bi in range(hist_bins):
        j = max(0, min(seg_k - (hist_bins - bi), n - 1))
        full = ru_gain_hist[j]
        cs = conn_hist[max(0, min(j, len(conn_hist) - 1))]
        for ru in STATE_RUS[int(cs)]:
            obs[ru - 1, bi] = full[ru - 1]
    oh = np.zeros(N_ACTIONS, np.float32)
    oh[int(conn_state)] = 1.0
    return np.concatenate([obs.reshape(-1), oh])


# -------------------------
# 비교 기준 (Final_Test_260217 의 LR / KF, GT 한 스텝만 문맥)
# -------------------------
def lr_predict(gt0: np.ndarray, future_len: int) -> np.ndarray:
    B, taps = gt0.shape
    n = np.arange(taps, dtype=np.float64)
    nm, nv = n.mean(), max(((n - n.mean()) ** 2).mean(), 1e-12)
    nf = np.arange(taps, taps + future_len, dtype=np.float64)[None]
    out = []
    for y in (gt0.real, gt0.imag):
        ym = y.mean(1, keepdims=True)
        a = ((n[None] - nm) * (y - ym)).mean(1, keepdims=True) / nv
        out.append(a * nf + (ym - a * nm))
    return out[0] + 1j * out[1]


def _kf_1d(y, future_len, r_rel, q_rel):
    B, taps = y.shape
    var = np.maximum(y.var(1), 1e-12)
    r = np.maximum(r_rel * var, 1e-12)
    q = np.maximum(q_rel * var, 1e-12)
    q11, q12, q22 = q / 4.0, q / 2.0, q
    x0, x1 = y[:, 0].copy(), np.zeros(B)
    p00, p01, p10, p11 = np.ones(B), np.zeros(B), np.zeros(B), np.ones(B)
    for k in range(taps):
        x0 = x0 + x1
        a00 = p00 + p10 + p01 + p11 + q11
        a01 = p01 + p11 + q12
        a10 = p10 + p11 + q12
        a11 = p11 + q22
        p00, p01, p10, p11 = a00, a01, a10, a11
        innov = y[:, k] - x0
        S = p00 + r
        K0, K1 = p00 / S, p10 / S
        x0, x1 = x0 + K0 * innov, x1 + K1 * innov
        p00, p01, p10, p11 = (1 - K0) * p00, (1 - K0) * p01, -K1 * p00 + p10, -K1 * p01 + p11
    out = []
    for _ in range(future_len):
        x0 = x0 + x1
        out.append(x0.copy())
    return np.stack(out, 1)


def kf_predict(gt0: np.ndarray, future_len: int, r_rel: float = 1e-2, q_rel: float = 1e-4) -> np.ndarray:
    return _kf_1d(gt0.real.astype(np.float64), future_len, r_rel, q_rel) + \
        1j * _kf_1d(gt0.imag.astype(np.float64), future_len, r_rel, q_rel)
