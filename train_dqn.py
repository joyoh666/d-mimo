# [YT 1019] 형주형 Final_Train_260316_capacity.ipynb 첫 셀을 그대로 옮긴 것. 바꾼 줄에만 [YT 1019] 표시.
#           cd live_demo 후 python train_dqn.py
#!/usr/bin/env python3
"""
Ridiculously Easy Double DQN RU Predictor (Small GRU)

Task definition:
- Observation at step k:
  - RU gains from -400ms to -100ms as 3 bins (9-dim = 3 RUs x 3 bins)
  - each bin masked by real connected RU at that historical segment
  - plus previous/current connection one-hot (6-dim)
  => total obs_dim = 15
- Action: choose one of RU1/RU2/RU3/RU12/RU13/RU23
- Reward at step k:
  selected RU power on segment k+1 minus tiny handover penalty
  reward = power(action, k+1) - 0.0005 * 1[action != prev_conn]

Model:
- GRUCell memory size = 8
- Similar/slightly larger parameter count than small MLP
"""

from __future__ import annotations

import os
import time
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


GPU_ID = 0  # [YT 1019]


# The action space stays fixed at three RUs. Antenna columns are assigned to
# these RUs in contiguous, near-equal groups (for example, 3 -> 1/1/1 and
# 6 -> 2/2/2).
NUM_RUS = 3


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{GPU_ID}")
    return torch.device("cpu")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return float(end)
    frac = min(max(step, 0) / float(duration), 1.0)
    return float(start + frac * (end - start))


STATE_NAMES = ["RU1", "RU2", "RU3", "RU12", "RU13", "RU23"]
N_ACTIONS = 6

STATE_RUS: Dict[int, List[int]] = {
    0: [1],
    1: [2],
    2: [3],
    3: [1, 2],
    4: [1, 3],
    5: [2, 3],
}


def build_ru_antenna_map(num_ants: int, num_rus: int = NUM_RUS) -> Dict[int, List[int]]:
    """Assign every antenna to one of the fixed RUs as evenly as possible."""
    num_ants = int(num_ants)
    num_rus = int(num_rus)
    if num_ants < num_rus:
        raise ValueError(
            f"NUM_ANTS={num_ants} must be >= NUM_RUS={num_rus}; "
            "each RU needs at least one antenna."
        )

    splits = np.array_split(np.arange(num_ants, dtype=np.int64), num_rus)
    return {ru + 1: split.tolist() for ru, split in enumerate(splits)}


class WindowStore:
    def __init__(self, window_dir: str):
        self.window_dir = Path(window_dir)
        x_path = self.window_dir / "X.npz"
        m_path = self.window_dir / "manifest.csv"
        if not x_path.is_file():
            raise FileNotFoundError(f"Missing: {x_path}")
        if not m_path.is_file():
            raise FileNotFoundError(f"Missing: {m_path}")

        self._npz = np.load(str(x_path), mmap_mode="r")
        if "X" not in self._npz:
            raise KeyError(f"'X' not found in {x_path}")
        self.X = self._npz["X"]
        if self.X.ndim != 3 or self.X.shape[-1] < NUM_RUS:
            raise ValueError(f"Expected X shape (N,L,num_ants) with num_ants >= {NUM_RUS}, got {self.X.shape}")

        self.N = int(self.X.shape[0])
        self.L = int(self.X.shape[1])
        self.num_ants = int(self.X.shape[2])
        self.ru_ants = build_ru_antenna_map(self.num_ants)
        self.dt = float(self._npz["dt"]) if "dt" in self._npz else 0.005
        self.window_sec = float(self._npz["window_sec"]) if "window_sec" in self._npz else (self.L * self.dt)
        self.gain_scales = (
            self._npz["gain_scales"].astype(np.float32)
            if "gain_scales" in self._npz
            else np.ones((self.num_ants,), np.float32)
        )
        if self.gain_scales.shape != (self.num_ants,):
            raise ValueError(
                f"Expected gain_scales shape ({self.num_ants},), got {self.gain_scales.shape}"
            )
        self.gain_match_enable = int(self._npz["gain_match_enable"]) if "gain_match_enable" in self._npz else 0

        self.manifest = pd.read_csv(str(m_path))
        if "sample_id" not in self.manifest.columns:
            self.manifest["sample_id"] = np.arange(self.N, dtype=int)
        self.manifest = self.manifest.sort_values("sample_id").reset_index(drop=True)
        if len(self.manifest) != self.N:
            raise ValueError(f"manifest rows ({len(self.manifest)}) != X windows N ({self.N})")

    def __len__(self) -> int:
        return self.N

    def get_window_gains(self, idx: int) -> np.ndarray:
        H = np.array(self.X[int(idx)], copy=False)
        g = np.abs(H).astype(np.float32)
        bad = ~np.isfinite(g)
        if np.any(bad):
            g[bad] = 0.0
        return g


@dataclass
class EnvConfig:
    seg_len_ticks: int = 20
    obs_bin_ticks: int = 20
    obs_hist_bins: int = 3
    mask_value: float = 0.0
    handover_penalty: float = 100000000
    inter_ru_phase_noise_std: float = math.pi / 8
    two_ru_penalty: float = 0.0
    noise_power: float = 1e-3
    reward_snr_db: float = 10.0
    max_episode_steps: int = 0


class RUWindowEnv:
    def __init__(self, store: WindowStore, episode_indices: List[int], cfg: EnvConfig, training: bool):
        self.store = store
        self.episode_indices = list(map(int, episode_indices))
        self.cfg = cfg
        self.training = bool(training)

        self.obs_dim = NUM_RUS * self.cfg.obs_hist_bins + N_ACTIONS
        self.n_actions = N_ACTIONS

        self._g: Optional[np.ndarray] = None
        self._ep_idx: int = -1
        self._seg_k: int = 0
        self._n_segs: int = 0
        self._steps: int = 0
        self._max_steps_this_ep: int = 0
        self._conn_state: int = 0
        self._conn_hist: List[int] = []

    def reset(self, ep_idx: Optional[int] = None) -> np.ndarray:
        if ep_idx is None:
            ep_idx = random.choice(self.episode_indices)
        ep_idx = int(ep_idx)
        if ep_idx not in self.episode_indices:
            raise ValueError("reset ep_idx not in this env's episode_indices")

        g = self.store.get_window_gains(ep_idx)
        L = g.shape[0]
        segL = int(self.cfg.seg_len_ticks)
        if (L % segL) != 0:
            raise ValueError(f"Window length L={L} not divisible by seg_len_ticks={segL}")

        n_segs = L // segL
        if n_segs < 2:
            raise ValueError(f"Window too short: n_segs={n_segs} < 2")

        self._g = g
        self._ep_idx = ep_idx
        self._seg_k = 0
        self._n_segs = int(n_segs)
        self._steps = 0
        self._conn_state = int(np.random.choice(list(range(N_ACTIONS))))
        self._conn_hist = [int(self._conn_state)]

        default_full = self._n_segs - 1
        if int(self.cfg.max_episode_steps) <= 0:
            self._max_steps_this_ep = default_full
        else:
            self._max_steps_this_ep = min(int(self.cfg.max_episode_steps), default_full)

        return self._make_obs()

    def _seg_slice(self, seg_idx: int) -> slice:
        segL = int(self.cfg.seg_len_ticks)
        a = seg_idx * segL
        b = a + segL
        return slice(a, b)

    def _ru_gain_from_segment(self, seg_idx: int) -> np.ndarray:
        assert self._g is not None
        seg = self._g[self._seg_slice(seg_idx), :]
        gains = []
        for ru in range(1, NUM_RUS + 1):
            ant_indices = self.store.ru_ants[ru]
            power_t = np.sum(seg[:, ant_indices] ** 2, axis=1)
            gains.append(float(np.sqrt(np.maximum(power_t, 0.0)).mean()))
        return np.asarray(gains, dtype=np.float32)

    def _all_state_power_from_segment(self, seg_idx: int) -> np.ndarray:
        assert self._g is not None
        seg = self._g[self._seg_slice(seg_idx), :].astype(np.float64)

        ru_power_t = {
            ru: np.sum(seg[:, self.store.ru_ants[ru]] ** 2, axis=1)
            for ru in range(1, NUM_RUS + 1)
        }

        phase_std = float(self.cfg.inter_ru_phase_noise_std)

        def pair_power(pa_t: np.ndarray, pb_t: np.ndarray) -> float:
            a = np.sqrt(np.maximum(pa_t, 0.0) * 0.5)
            b = np.sqrt(np.maximum(pb_t, 0.0) * 0.5)
            dphi = np.random.normal(0.0, phase_std, size=a.shape)
            p = a * a + b * b + 2.0 * a * b * np.cos(dphi)
            return float(np.mean(p))

        state_power = []
        for state_idx in range(N_ACTIONS):
            rus = STATE_RUS[state_idx]
            if len(rus) == 1:
                state_power.append(float(np.mean(ru_power_t[rus[0]])))
            elif len(rus) == 2:
                state_power.append(pair_power(ru_power_t[rus[0]], ru_power_t[rus[1]]))
            else:
                raise ValueError(f"Unsupported RU state: {rus}")

        power = np.asarray(state_power, dtype=np.float32)
        # Anchor: noise_power=1e-3 at 10 dB; scale by SNR only for rate reward calculation.
        noise_power_reward = float(self.cfg.noise_power) * (10.0 ** ((10.0 - float(self.cfg.reward_snr_db)) / 10.0))
        return np.log2(1.0 + power / noise_power_reward)

    def _make_obs(self) -> np.ndarray:
        B = int(self.cfg.obs_hist_bins)
        obs = np.full((NUM_RUS, B), float(self.cfg.mask_value), dtype=np.float32)

        # Observation bins: [k-3, k-2, k-1]  => -400ms to -100ms
        for bi in range(B):
            seg_idx = int(self._seg_k) - (B - bi)
            seg_idx = max(0, min(seg_idx, self._n_segs - 1))
            full = self._ru_gain_from_segment(seg_idx).astype(np.float32)

            hist_idx = max(0, min(seg_idx, len(self._conn_hist) - 1))
            conn_state_seg = int(self._conn_hist[hist_idx])
            for ru in STATE_RUS[conn_state_seg]:
                obs[ru - 1, bi] = full[ru - 1]

        conn_oh = np.zeros((N_ACTIONS,), dtype=np.float32)
        conn_oh[int(self._conn_state)] = 1.0
        return np.concatenate([obs.reshape(-1).astype(np.float32), conn_oh], axis=0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self._g is not None
        action = int(np.clip(int(action), 0, N_ACTIONS - 1))

        obs_seg = int(self._seg_k)
        conn_before = int(self._conn_state)
        rew_seg = obs_seg + 1

        state_power = self._all_state_power_from_segment(rew_seg)
        handover_attempt = 1 if (int(action) != int(conn_before)) else 0
        reward = (
            float(state_power[action])
            - float(self.cfg.handover_penalty) * float(handover_attempt)
            - float(self.cfg.two_ru_penalty) * float(action >= NUM_RUS)
            - float(3.0)
        )

        self._conn_state = int(action)
        self._seg_k += 1
        self._steps += 1

        if len(self._conn_hist) <= int(self._seg_k):
            self._conn_hist.append(int(self._conn_state))

        done = (self._steps >= self._max_steps_this_ep)
        obs = self._make_obs()

        info = {
            "window_index": int(self._ep_idx),
            "dataset": str(self.store.manifest.loc[int(self._ep_idx), "dataset"]) if "dataset" in self.store.manifest.columns else "unknown",
            "seg_k": int(self._seg_k),
            "obs_seg": int(obs_seg),
            "reward_seg": int(rew_seg),
            "action": int(action),
            "reward": float(reward),
            "selected_state_power": float(state_power[action]),
            "best_state_power": float(np.max(state_power)),
            "is_best_action": int(action == int(np.argmax(state_power))),
            "handover_attempt": int(handover_attempt),
            "conn_state_before": int(conn_before),
            "conn_state_after": int(self._conn_state),
        }
        return obs, reward, done, info


class VecEnv:
    def __init__(self, envs: List[RUWindowEnv]):
        self.envs = envs
        self.n = len(envs)
        self.obs_dim = envs[0].obs_dim
        self.n_actions = envs[0].n_actions

    def reset(self) -> np.ndarray:
        obs = [e.reset() for e in self.envs]
        return np.stack(obs, axis=0)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for i, e in enumerate(self.envs):
            o, r, d, info = e.step(int(actions[i]))
            if d:
                o = e.reset()
                info["episode_reset"] = True
            else:
                info["episode_reset"] = False
            obs_list.append(o)
            rew_list.append(r)
            done_list.append(d)
            info_list.append(info)
        return (
            np.stack(obs_list, axis=0).astype(np.float32),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=np.bool_),
            info_list,
        )


class GRUQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.n_actions = int(n_actions)

        self.gru = nn.GRUCell(input_size=self.obs_dim, hidden_size=self.hidden_size)
        self.q_head = nn.Linear(self.hidden_size, self.n_actions)
        self.h0 = nn.Parameter(torch.zeros(self.hidden_size))
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        for n, p in self.gru.named_parameters():
            if "weight" in n:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in n:
                nn.init.constant_(p, 0.0)

    def init_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.h0.view(1, -1).expand(batch, -1).to(device)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_next = self.gru(obs, h)
        q = self.q_head(h_next)
        return q, h_next


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, hidden_size: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.h = np.zeros((self.capacity, self.hidden_size), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_h = np.zeros((self.capacity, self.hidden_size), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, obs, h, actions, rewards, next_obs, next_h, dones):
        n = int(actions.shape[0])
        idxs = (np.arange(n) + self.pos) % self.capacity

        self.obs[idxs] = np.asarray(obs, dtype=np.float32)
        self.h[idxs] = np.asarray(h, dtype=np.float32)
        self.actions[idxs] = np.asarray(actions, dtype=np.int64)
        self.rewards[idxs] = np.asarray(rewards, dtype=np.float32)
        self.next_obs[idxs] = np.asarray(next_obs, dtype=np.float32)
        self.next_h[idxs] = np.asarray(next_h, dtype=np.float32)
        self.dones[idxs] = np.asarray(dones, dtype=np.float32)

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return (
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.h[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_obs[idx]).to(device),
            torch.from_numpy(self.next_h[idx]).to(device),
            torch.from_numpy(self.dones[idx]).to(device),
        )


@dataclass
class DQNConfig:
    total_env_steps: int = 1_500_000
    num_envs: int = 16
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 50_000
    learning_starts: int = 2_000
    train_every: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 2_000
    max_grad_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 150_000


@torch.no_grad()
def evaluate_on_windows(
    q_net: GRUQNetwork,
    store: WindowStore,
    test_indices: List[int],
    env_cfg: EnvConfig,
    device: torch.device,
    max_eval_episodes: int = 200,
) -> Dict[str, float]:
    q_net.eval()

    if len(test_indices) == 0:
        return {
            "eval_reward_mean": float("nan"),
            "eval_best_action_ratio": float("nan"),
            "eval_handover_attempt_per_step": float("nan"),
            "n_episodes": 0,
        }

    idxs = list(test_indices)
    random.shuffle(idxs)
    if max_eval_episodes > 0:
        idxs = idxs[:max_eval_episodes]

    returns, best_ratios, ho_rates = [], [], []
    for ep_idx in idxs:
        env = RUWindowEnv(store, [ep_idx], env_cfg, training=False)
        obs = env.reset(ep_idx=ep_idx)
        h = q_net.init_hidden(1, device)

        ep_ret, ep_best, ep_ho, steps = 0.0, 0.0, 0.0, 0
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).to(device).view(1, -1)
            q_values, h = q_net(obs_t, h)
            act = int(torch.argmax(q_values, dim=-1).item())

            obs, r, done, info = env.step(act)
            ep_ret += float(r)
            ep_best += float(info["is_best_action"])
            ep_ho += float(info["handover_attempt"])
            steps += 1

        returns.append(ep_ret / max(steps, 1))
        best_ratios.append(ep_best / max(steps, 1))
        ho_rates.append(ep_ho / max(steps, 1))

    return {
        "eval_reward_mean": float(np.mean(returns)) if returns else float("nan"),
        "eval_best_action_ratio": float(np.mean(best_ratios)) if best_ratios else float("nan"),
        "eval_handover_attempt_per_step": float(np.mean(ho_rates)) if ho_rates else float("nan"),
        "n_episodes": int(len(returns)),
    }


def main(argv: Optional[List[str]] = None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--window-dir", type=str, default="./data_ready/dqn_ws4p00s")  # [YT 1019]
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--total-env-steps", type=int, default=1_500_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--replay-size", type=int, default=50_000)
    ap.add_argument("--learning-starts", type=int, default=2_000)
    ap.add_argument("--train-every", type=int, default=1)
    ap.add_argument("--gradient-steps", type=int, default=1)
    ap.add_argument("--target-update-interval", type=int, default=2_000)
    ap.add_argument("--max-grad-norm", type=float, default=10.0)
    ap.add_argument("--two-ru-penalty", type=float, default=2.0)
    ap.add_argument("--reward-snr-db", type=float, default=10.0)
    ap.add_argument("--handover-penalty", type=float, default=5e-1)  # [YT 1019] 원래 아래 EnvConfig 에 5e-1 고정. 독스트링·Test 기본값은 5e-4
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-end", type=float, default=0.02)
    ap.add_argument("--epsilon-decay-steps", type=int, default=150_000)

    ap.add_argument("--hidden-size", type=int, default=8)

    ap.add_argument("--log-every", type=int, default=5_000)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--eval-max-episodes", type=int, default=200)
    ap.add_argument("--save-dir", type=str, default="./checkpoints_double_dqn_ru_windows4s_easiest")
    ap.add_argument("--save-every", type=int, default=50_000)

    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print("[ARGS] Ignoring unknown args (likely Jupyter):", unknown)

    seed_all(args.seed)
    device = get_device()
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[CUDA] GPU_ID={GPU_ID} | name: {torch.cuda.get_device_name(device)}")

    store = WindowStore(args.window_dir)
    print(f"[DATA] Loaded windows from: {Path(args.window_dir).resolve()}")
    print(
        f"[DATA] X shape: (N,L,{store.num_ants}) = {store.X.shape} "
        f"| dt={store.dt} | window_sec={store.window_sec}"
    )
    print(f"[DATA] RU antenna map: {store.ru_ants}")

    if "split" not in store.manifest.columns:
        raise ValueError("manifest.csv must contain an episode-level 'split' column")
    split_values = store.manifest["split"].astype(str).to_numpy()
    train_indices = np.flatnonzero(split_values == "train").tolist()
    validation_indices = np.flatnonzero(split_values == "validation").tolist()
    test_indices = np.flatnonzero(split_values == "test").tolist()
    if min(len(train_indices), len(validation_indices), len(test_indices)) == 0:
        raise ValueError("train, validation, and test splits must all be non-empty")
    print(
        f"[SPLIT] windows N={len(store)} => train={len(train_indices)} "
        f"validation={len(validation_indices)} test={len(test_indices)}"
    )

    env_cfg = EnvConfig(seg_len_ticks=20, obs_bin_ticks=20, obs_hist_bins=3, mask_value=0.0, handover_penalty=float(args.handover_penalty), inter_ru_phase_noise_std=math.pi / 8.0, two_ru_penalty=float(args.two_ru_penalty), noise_power=1e-3, reward_snr_db=float(args.reward_snr_db), max_episode_steps=0)  # [YT 1019] handover_penalty 인자화
    dqn_cfg = DQNConfig(
        total_env_steps=int(args.total_env_steps),
        num_envs=int(args.num_envs),
        gamma=float(args.gamma),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        replay_size=int(args.replay_size),
        learning_starts=int(args.learning_starts),
        train_every=int(args.train_every),
        gradient_steps=int(args.gradient_steps),
        target_update_interval=int(args.target_update_interval),
        max_grad_norm=float(args.max_grad_norm),
        epsilon_start=float(args.epsilon_start),
        epsilon_end=float(args.epsilon_end),
        epsilon_decay_steps=int(args.epsilon_decay_steps),
    )

    train_envs = [RUWindowEnv(store, train_indices, env_cfg, training=True) for _ in range(dqn_cfg.num_envs)]
    venv = VecEnv(train_envs)
    obs = venv.reset()

    q_net = GRUQNetwork(obs_dim=venv.obs_dim, hidden_size=int(args.hidden_size), n_actions=venv.n_actions).to(device)
    target_net = GRUQNetwork(obs_dim=venv.obs_dim, hidden_size=int(args.hidden_size), n_actions=venv.n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    n_params = count_trainable_params(q_net)
    print(
        f"[MODEL] obs_dim={venv.obs_dim} hidden_size={q_net.hidden_size} "
        f"n_actions={q_net.n_actions} trainable_params={n_params}"
    )
    print("[TASK] Obs=masked RU gains from -400~-100ms (9D) + prev-conn onehot (3D), Reward=power@(k+1)-0.0005*handover.")

    optim = torch.optim.Adam(q_net.parameters(), lr=dqn_cfg.lr, eps=1e-5)
    replay = ReplayBuffer(capacity=dqn_cfg.replay_size, obs_dim=venv.obs_dim, hidden_size=q_net.hidden_size)
    os.makedirs(args.save_dir, exist_ok=True)

    h_env = q_net.init_hidden(dqn_cfg.num_envs, device).detach().cpu().numpy().astype(np.float32)

    ep_ret_running = np.zeros((dqn_cfg.num_envs,), dtype=np.float64)
    ep_best_running = np.zeros((dqn_cfg.num_envs,), dtype=np.float64)
    ep_ho_running = np.zeros((dqn_cfg.num_envs,), dtype=np.float64)
    ep_len_running = np.zeros((dqn_cfg.num_envs,), dtype=np.int64)

    recent_returns = deque(maxlen=100)
    recent_best = deque(maxlen=100)
    recent_handover = deque(maxlen=100)
    recent_losses = deque(maxlen=200)

    global_step = 0
    env_iter = 0
    t_start = time.time()
    last_target_sync = 0
    next_log_step = int(args.log_every)
    next_eval_step = int(args.eval_every)
    next_save_step = int(args.save_every)

    while global_step < dqn_cfg.total_env_steps:
        q_net.train()

        epsilon = linear_schedule(global_step, dqn_cfg.epsilon_start, dqn_cfg.epsilon_end, dqn_cfg.epsilon_decay_steps)

        obs_t = torch.from_numpy(obs).to(device)
        h_t = torch.from_numpy(h_env).to(device)
        with torch.no_grad():
            q_values, h_next_t = q_net(obs_t, h_t)
            greedy_actions = torch.argmax(q_values, dim=-1).cpu().numpy()

        random_actions = np.random.randint(0, venv.n_actions, size=dqn_cfg.num_envs, dtype=np.int64)
        explore_mask = (np.random.rand(dqn_cfg.num_envs) < epsilon)
        actions = np.where(explore_mask, random_actions, greedy_actions).astype(np.int64)

        next_obs, rewards, dones, infos = venv.step(actions)

        h_next_np = h_next_t.detach().cpu().numpy().astype(np.float32)
        # Reset hidden state for envs that auto-reset episode
        for i in range(dqn_cfg.num_envs):
            if bool(infos[i].get("episode_reset", False)):
                h_next_np[i, :] = 0.0

        for i in range(dqn_cfg.num_envs):
            ep_ret_running[i] += float(rewards[i])
            ep_best_running[i] += float(infos[i]["is_best_action"])
            ep_ho_running[i] += float(infos[i]["handover_attempt"])
            ep_len_running[i] += 1
            if dones[i]:
                recent_returns.append(float(ep_ret_running[i] / max(ep_len_running[i], 1)))
                recent_best.append(float(ep_best_running[i] / max(ep_len_running[i], 1)))
                recent_handover.append(float(ep_ho_running[i] / max(ep_len_running[i], 1)))
                ep_ret_running[i] = 0.0
                ep_best_running[i] = 0.0
                ep_ho_running[i] = 0.0
                ep_len_running[i] = 0

        replay.add_batch(obs, h_env, actions, rewards, next_obs, h_next_np, dones.astype(np.float32))

        obs = next_obs
        h_env = h_next_np
        env_iter += 1
        global_step += dqn_cfg.num_envs

        if (global_step >= dqn_cfg.learning_starts) and (len(replay) >= dqn_cfg.batch_size) and (env_iter % dqn_cfg.train_every == 0):
            for _ in range(dqn_cfg.gradient_steps):
                b_obs, b_h, b_act, b_rew, b_next_obs, b_next_h, b_done = replay.sample(dqn_cfg.batch_size, device)

                q_pred_all, _ = q_net(b_obs, b_h)
                q_pred = q_pred_all.gather(1, b_act.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    next_online_q, _ = q_net(b_next_obs, b_next_h)
                    next_actions = next_online_q.argmax(dim=-1, keepdim=True)
                    next_target_q, _ = target_net(b_next_obs, b_next_h)
                    next_q = next_target_q.gather(1, next_actions).squeeze(1)
                    target = b_rew + dqn_cfg.gamma * (1.0 - b_done) * next_q

                loss = F.smooth_l1_loss(q_pred, target)
                recent_losses.append(float(loss.item()))

                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), dqn_cfg.max_grad_norm)
                optim.step()

        if (global_step - last_target_sync) >= dqn_cfg.target_update_interval:
            target_net.load_state_dict(q_net.state_dict())
            last_target_sync = global_step

        if global_step >= next_log_step:
            elapsed_min = (time.time() - t_start) / 60.0
            rew_mean = float(np.mean(recent_returns)) if recent_returns else float("nan")
            best_mean = float(np.mean(recent_best)) if recent_best else float("nan")
            ho_mean = float(np.mean(recent_handover)) if recent_handover else float("nan")
            loss_mean = float(np.mean(recent_losses)) if recent_losses else float("nan")
            print(
                f"[step {global_step:9d}] eps={epsilon:.3f} buffer={len(replay):6d} loss={loss_mean:.4f} "
                f"train_reward/step={rew_mean:.4f} train_best_action_ratio={best_mean:.4f} train_handover/step={ho_mean:.5f} | t={elapsed_min:.1f} min"
            )
            next_log_step += int(args.log_every)

        if (args.eval_every > 0) and (global_step >= next_eval_step):
            m = evaluate_on_windows(q_net=q_net, store=store, test_indices=validation_indices, env_cfg=env_cfg, device=device, max_eval_episodes=int(args.eval_max_episodes))
            elapsed_min = (time.time() - t_start) / 60.0
            print(
                f"[VALIDATION step {global_step:9d}] reward/step={m['eval_reward_mean']:.4f} "
                f"best_action_ratio={m['eval_best_action_ratio']:.4f} handover/step={m['eval_handover_attempt_per_step']:.5f} "
                f"n_ep={m['n_episodes']} | t={elapsed_min:.1f} min"
            )
            next_eval_step += int(args.eval_every)

        if (args.save_every > 0) and (global_step >= next_save_step):
            ckpt = os.path.join(args.save_dir, f"double_dqn_ru_easiest_step{global_step:09d}.pt")
            torch.save(
                {
                    "global_step": global_step,
                    "env_iter": env_iter,
                    "q_net": q_net.state_dict(),
                    "target_net": target_net.state_dict(),
                    "optim": optim.state_dict(),
                    "env_cfg": env_cfg.__dict__,
                    "dqn_cfg": dqn_cfg.__dict__,
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                    "test_indices": test_indices,
                    "window_dir": str(Path(args.window_dir).resolve()),
                    "gain_scales_in_xnpz": store.gain_scales,
                    "gain_match_enable": store.gain_match_enable,
                    "num_ants": store.num_ants,
                    "num_rus": NUM_RUS,
                    "ru_ants": store.ru_ants,
                    "args": vars(args),
                    "GPU_ID": GPU_ID,
                    "model_meta": {
                        "obs_dim": venv.obs_dim,
                        "hidden_size": q_net.hidden_size,
                        "n_actions": q_net.n_actions,
                        "num_ants": store.num_ants,
                        "num_rus": NUM_RUS,
                        "ru_ants": store.ru_ants,
                        "n_params": n_params,
                        "algo": "double_dqn_easiest_gru",
                        "task_alignment": "obs[k-3,k-2,k-1] masked + conn_onehot -> reward_power@k+1 - 0.0005*handover",
                    },
                },
                ckpt,
            )
            print(f"[SAVE] {ckpt}")
            next_save_step += int(args.save_every)

    test_metrics = evaluate_on_windows(
        q_net=q_net, store=store, test_indices=test_indices, env_cfg=env_cfg,
        device=device, max_eval_episodes=0,
    )
    print(
        f"[FINAL TEST] reward/step={test_metrics['eval_reward_mean']:.4f} "
        f"best_action_ratio={test_metrics['eval_best_action_ratio']:.4f} "
        f"handover/step={test_metrics['eval_handover_attempt_per_step']:.5f} "
        f"n_ep={test_metrics['n_episodes']}"
    )
    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
