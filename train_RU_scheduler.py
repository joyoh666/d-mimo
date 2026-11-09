#!/usr/bin/env python3
"""Train the GRU Double-DQN RU scheduler on prepared dataset splits.

This file intentionally does not generate or split data. It reads the prepared
``scheduler/{train,validation,test}`` directories produced by
``scripts/prepare_learning_datasets.py``.

The antenna layout is read from ``ru_antenna_counts`` in each X.npz file. For
example, [1, 1, 1] means one antenna/channel per RU and [2, 2, 2] means two per
RU. Channels are never copied or padded. The action space contains every
single-RU action followed by every two-RU combination.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"gpu_id={gpu_id} is invalid; visible CUDA devices: "
            f"0..{torch.cuda.device_count() - 1}"
        )
    return torch.device(f"cuda:{gpu_id}")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_schedule(step: int, start: float, end: float, duration: int) -> float:
    if duration <= 0:
        return float(end)
    frac = min(max(step, 0) / float(duration), 1.0)
    return float(start + frac * (end - start))


def state_layout(num_rus: int) -> Tuple[Tuple[int, ...], ...]:
    """Return single-RU states first, then all two-RU states (zero based)."""
    if num_rus < 1:
        raise ValueError(f"num_rus must be positive, got {num_rus}")
    singles = [(i,) for i in range(num_rus)]
    pairs = list(itertools.combinations(range(num_rus), 2))
    return tuple(singles + pairs)


def state_name(state: Sequence[int]) -> str:
    return "+".join(f"RU{i + 1}" for i in state)


class WindowStore:
    """A prepared scheduler split and its dynamic RU/antenna layout."""

    def __init__(self, window_dir: Path | str):
        self.window_dir = Path(window_dir)
        x_path = self.window_dir / "X.npz"
        manifest_path = self.window_dir / "manifest.csv"
        if not x_path.is_file():
            raise FileNotFoundError(f"Missing: {x_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing: {manifest_path}")

        self._npz = np.load(str(x_path), mmap_mode="r", allow_pickle=False)
        if "X" not in self._npz:
            raise KeyError(f"'X' not found in {x_path}")
        self.X = self._npz["X"]
        if self.X.ndim != 3:
            raise ValueError(f"Expected X shape (N,L,C), got {self.X.shape}")

        self.N, self.L, self.num_antennas = map(int, self.X.shape)
        if self.N < 1 or self.L < 1 or self.num_antennas < 1:
            raise ValueError(f"X must have non-zero dimensions, got {self.X.shape}")

        if "num_antennas" in self._npz:
            meta_num_antennas = int(self._npz["num_antennas"])
            if meta_num_antennas != self.num_antennas:
                raise ValueError(
                    f"num_antennas metadata={meta_num_antennas}, but X has "
                    f"C={self.num_antennas}: {x_path}"
                )

        if "ru_antenna_counts" in self._npz:
            counts = np.asarray(self._npz["ru_antenna_counts"], dtype=np.int64).reshape(-1)
        else:
            # Safe fallback for older one-channel-per-RU datasets.
            counts = np.ones((self.num_antennas,), dtype=np.int64)
        if counts.size < 1 or np.any(counts <= 0):
            raise ValueError(f"ru_antenna_counts must contain positive integers, got {counts}")
        if int(counts.sum()) != self.num_antennas:
            raise ValueError(
                f"sum(ru_antenna_counts)={int(counts.sum())} does not match "
                f"X.shape[-1]={self.num_antennas}: {x_path}"
            )

        self.ru_antenna_counts = tuple(int(x) for x in counts)
        self.num_rus = len(self.ru_antenna_counts)
        boundaries = np.concatenate([[0], np.cumsum(counts)]).astype(int)
        self.ru_slices = tuple(
            slice(int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(self.num_rus)
        )
        self.action_states = state_layout(self.num_rus)
        self.action_names = tuple(state_name(s) for s in self.action_states)
        self.n_actions = len(self.action_states)

        self.dt = float(self._npz["dt"]) if "dt" in self._npz else 0.005
        self.measured_dt = (
            float(self._npz["measured_dt"])
            if "measured_dt" in self._npz
            else self.dt
        )
        self.window_sec = (
            float(self._npz["window_sec"])
            if "window_sec" in self._npz
            else self.L * self.dt
        )
        self.scale = float(self._npz["scale"]) if "scale" in self._npz else 1.0
        self.scenario_ids = (
            tuple(int(x) for x in np.asarray(self._npz["scenario_ids"]).reshape(-1))
            if "scenario_ids" in self._npz
            else tuple()
        )

        with manifest_path.open(newline="", encoding="utf-8") as f:
            self.manifest = list(csv.DictReader(f))
        if len(self.manifest) != self.N:
            raise ValueError(
                f"manifest rows ({len(self.manifest)}) != X windows N ({self.N}): "
                f"{manifest_path}"
            )

        sample_ids = [int(row.get("sample_id", i)) for i, row in enumerate(self.manifest)]
        order = np.argsort(sample_ids)
        if not np.array_equal(order, np.arange(self.N)):
            self.manifest = [self.manifest[int(i)] for i in order]
        sorted_ids = [int(row.get("sample_id", i)) for i, row in enumerate(self.manifest)]
        if sorted_ids != list(range(self.N)):
            raise ValueError(
                f"manifest sample_id must be exactly 0..{self.N - 1}: {manifest_path}"
            )

    def __len__(self) -> int:
        return self.N

    def get_window_gains(self, idx: int) -> np.ndarray:
        gains = np.abs(np.asarray(self.X[int(idx)])).astype(np.float32)
        gains[~np.isfinite(gains)] = 0.0
        return gains

    def row_label(self, idx: int) -> str:
        row = self.manifest[int(idx)]
        return str(row.get("dataset", row.get("scenario_id", "unknown")))


def validate_split_compatibility(stores: Dict[str, WindowStore]) -> None:
    train = stores["train"]
    for split, store in stores.items():
        if store.L != train.L:
            raise ValueError(f"{split} L={store.L}, train L={train.L}")
        if store.num_antennas != train.num_antennas:
            raise ValueError(
                f"{split} num_antennas={store.num_antennas}, "
                f"train={train.num_antennas}"
            )
        if store.ru_antenna_counts != train.ru_antenna_counts:
            raise ValueError(
                f"{split} ru_antenna_counts={store.ru_antenna_counts}, "
                f"train={train.ru_antenna_counts}"
            )
        if not math.isclose(store.dt, train.dt, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{split} dt={store.dt}, train dt={train.dt}")
        if not math.isclose(store.scale, train.scale, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError(f"{split} scale={store.scale}, train scale={train.scale}")

    scenario_sets = {
        split: set(store.scenario_ids) for split, store in stores.items()
    }
    for a, b in itertools.combinations(stores.keys(), 2):
        overlap = scenario_sets[a] & scenario_sets[b]
        if overlap:
            raise ValueError(f"Scenario leakage between {a} and {b}: {sorted(overlap)}")


@dataclass
class EnvConfig:
    seg_len_ticks: int
    obs_hist_bins: int = 3
    mask_value: float = 0.0
    handover_penalty: float = 0.05
    inter_ru_phase_noise_std: float = math.pi / 8.0
    two_ru_penalty: float = 0.1
    noise_power: float = 1e-3
    reward_snr_db: float = 10.0
    reward_offset: float = 3.0
    max_episode_steps: int = 0


class RUWindowEnv:
    def __init__(
        self,
        store: WindowStore,
        episode_indices: Sequence[int],
        cfg: EnvConfig,
        training: bool,
    ):
        self.store = store
        self.episode_indices = list(map(int, episode_indices))
        if not self.episode_indices:
            raise ValueError("episode_indices cannot be empty")
        self.cfg = cfg
        self.training = bool(training)

        self.num_rus = store.num_rus
        self.action_states = store.action_states
        self.action_names = store.action_names
        self.n_actions = store.n_actions
        self.obs_dim = self.num_rus * self.cfg.obs_hist_bins + self.n_actions

        self._g: Optional[np.ndarray] = None
        self._ep_idx = -1
        self._seg_k = 0
        self._n_segs = 0
        self._steps = 0
        self._max_steps_this_ep = 0
        self._conn_state = 0
        self._conn_hist: List[int] = []

    def reset(self, ep_idx: Optional[int] = None) -> np.ndarray:
        if ep_idx is None:
            ep_idx = random.choice(self.episode_indices)
        ep_idx = int(ep_idx)
        if ep_idx not in self.episode_indices:
            raise ValueError("reset ep_idx not in this env's episode_indices")

        gains = self.store.get_window_gains(ep_idx)
        seg_len = int(self.cfg.seg_len_ticks)
        if seg_len <= 0 or gains.shape[0] % seg_len != 0:
            raise ValueError(
                f"Window length L={gains.shape[0]} is not divisible by "
                f"seg_len_ticks={seg_len}"
            )
        n_segs = gains.shape[0] // seg_len
        if n_segs < 2:
            raise ValueError(f"Window too short: n_segs={n_segs} < 2")

        self._g = gains
        self._ep_idx = ep_idx
        self._seg_k = 0
        self._n_segs = int(n_segs)
        self._steps = 0
        self._conn_state = int(np.random.randint(self.n_actions))
        self._conn_hist = [self._conn_state]

        full_steps = self._n_segs - 1
        if self.cfg.max_episode_steps <= 0:
            self._max_steps_this_ep = full_steps
        else:
            self._max_steps_this_ep = min(self.cfg.max_episode_steps, full_steps)
        return self._make_obs()

    def _seg_slice(self, seg_idx: int) -> slice:
        start = int(seg_idx) * self.cfg.seg_len_ticks
        return slice(start, start + self.cfg.seg_len_ticks)

    def _ru_power_series(self, seg_idx: int) -> np.ndarray:
        """Return per-tick power for every RU: shape (ticks, num_rus)."""
        assert self._g is not None
        segment = self._g[self._seg_slice(seg_idx), :].astype(np.float64)
        powers = [
            np.sum(segment[:, channel_slice] ** 2, axis=1)
            for channel_slice in self.store.ru_slices
        ]
        return np.stack(powers, axis=1)

    def _ru_gain_from_segment(self, seg_idx: int) -> np.ndarray:
        ru_power_t = self._ru_power_series(seg_idx)
        return np.sqrt(np.maximum(ru_power_t, 0.0)).mean(axis=0).astype(np.float32)

    def _all_state_rate_from_segment(self, seg_idx: int) -> np.ndarray:
        ru_power_t = self._ru_power_series(seg_idx)
        phase_std = float(self.cfg.inter_ru_phase_noise_std)
        state_powers: List[float] = []

        for state in self.action_states:
            if len(state) == 1:
                state_powers.append(float(np.mean(ru_power_t[:, state[0]])))
                continue

            # The action space currently contains pairs only. Split the total
            # transmit power equally across the two selected RUs, matching the
            # behavior of the original six-channel implementation.
            ru_a, ru_b = state
            a = np.sqrt(np.maximum(ru_power_t[:, ru_a], 0.0) * 0.5)
            b = np.sqrt(np.maximum(ru_power_t[:, ru_b], 0.0) * 0.5)
            delta_phase = np.random.normal(0.0, phase_std, size=a.shape)
            combined = a * a + b * b + 2.0 * a * b * np.cos(delta_phase)
            state_powers.append(float(np.mean(combined)))

        power = np.asarray(state_powers, dtype=np.float32)
        noise_power_reward = self.cfg.noise_power * (
            10.0 ** ((10.0 - self.cfg.reward_snr_db) / 10.0)
        )
        return np.log2(1.0 + power / noise_power_reward)

    def _make_obs(self) -> np.ndarray:
        bins = int(self.cfg.obs_hist_bins)
        obs = np.full(
            (self.num_rus, bins), self.cfg.mask_value, dtype=np.float32
        )

        # Relative to the reward segment k+1, these are the B historical bins
        # ending one segment before the action segment.
        for bin_idx in range(bins):
            seg_idx = self._seg_k - (bins - bin_idx)
            seg_idx = max(0, min(seg_idx, self._n_segs - 1))
            full_ru_gains = self._ru_gain_from_segment(seg_idx)
            hist_idx = max(0, min(seg_idx, len(self._conn_hist) - 1))
            connected_action = self._conn_hist[hist_idx]
            for ru_idx in self.action_states[connected_action]:
                obs[ru_idx, bin_idx] = full_ru_gains[ru_idx]

        connection_one_hot = np.zeros((self.n_actions,), dtype=np.float32)
        connection_one_hot[self._conn_state] = 1.0
        return np.concatenate([obs.reshape(-1), connection_one_hot])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self._g is not None
        action = int(np.clip(int(action), 0, self.n_actions - 1))
        obs_seg = self._seg_k
        conn_before = self._conn_state
        reward_seg = obs_seg + 1

        state_rates = self._all_state_rate_from_segment(reward_seg)
        handover_attempt = int(action != conn_before)
        uses_two_rus = int(len(self.action_states[action]) == 2)
        reward = (
            float(state_rates[action])
            - self.cfg.handover_penalty * handover_attempt
            - self.cfg.two_ru_penalty * uses_two_rus
            - self.cfg.reward_offset
        )

        self._conn_state = action
        self._seg_k += 1
        self._steps += 1
        if len(self._conn_hist) <= self._seg_k:
            self._conn_hist.append(self._conn_state)

        done = self._steps >= self._max_steps_this_ep
        obs = self._make_obs()
        info = {
            "window_index": self._ep_idx,
            "dataset": self.store.row_label(self._ep_idx),
            "seg_k": self._seg_k,
            "obs_seg": obs_seg,
            "reward_seg": reward_seg,
            "action": action,
            "action_name": self.action_names[action],
            "reward": reward,
            "selected_state_rate": float(state_rates[action]),
            "best_state_rate": float(np.max(state_rates)),
            "is_best_action": int(action == int(np.argmax(state_rates))),
            "handover_attempt": handover_attempt,
            "conn_state_before": conn_before,
            "conn_state_after": self._conn_state,
        }
        return obs, reward, done, info


class VecEnv:
    def __init__(self, envs: Sequence[RUWindowEnv]):
        if not envs:
            raise ValueError("envs cannot be empty")
        self.envs = list(envs)
        self.n = len(self.envs)
        self.obs_dim = self.envs[0].obs_dim
        self.n_actions = self.envs[0].n_actions

    def reset(self) -> np.ndarray:
        return np.stack([env.reset() for env in self.envs], axis=0)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(int(action))
            if done:
                obs = env.reset()
                info["episode_reset"] = True
            else:
                info["episode_reset"] = False
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return (
            np.stack(obs_list).astype(np.float32),
            np.asarray(reward_list, dtype=np.float32),
            np.asarray(done_list, dtype=np.bool_),
            info_list,
        )


class GRUQNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.n_actions = int(n_actions)
        self.gru = nn.GRUCell(self.obs_dim, self.hidden_size)
        self.q_head = nn.Linear(self.hidden_size, self.n_actions)
        self._init_params()

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(parameter, 0.0)

    def init_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch, self.hidden_size), device=device)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_hidden = self.gru(obs, hidden)
        return self.q_head(next_hidden), next_hidden


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, hidden_size: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.hidden = np.zeros((capacity, hidden_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_hidden = np.zeros((capacity, hidden_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self, obs, hidden, actions, rewards, next_obs, next_hidden, dones
    ) -> None:
        n = int(actions.shape[0])
        indices = (np.arange(n) + self.pos) % self.capacity
        self.obs[indices] = np.asarray(obs, dtype=np.float32)
        self.hidden[indices] = np.asarray(hidden, dtype=np.float32)
        self.actions[indices] = np.asarray(actions, dtype=np.int64)
        self.rewards[indices] = np.asarray(rewards, dtype=np.float32)
        self.next_obs[indices] = np.asarray(next_obs, dtype=np.float32)
        self.next_hidden[indices] = np.asarray(next_hidden, dtype=np.float32)
        self.dones[indices] = np.asarray(dones, dtype=np.float32)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return (
            torch.from_numpy(self.obs[indices]).to(device),
            torch.from_numpy(self.hidden[indices]).to(device),
            torch.from_numpy(self.actions[indices]).to(device),
            torch.from_numpy(self.rewards[indices]).to(device),
            torch.from_numpy(self.next_obs[indices]).to(device),
            torch.from_numpy(self.next_hidden[indices]).to(device),
            torch.from_numpy(self.dones[indices]).to(device),
        )


@dataclass
class DQNConfig:
    total_env_steps: int
    num_envs: int
    gamma: float
    lr: float
    batch_size: int
    replay_size: int
    learning_starts: int
    train_every: int
    gradient_steps: int
    target_update_interval: int
    max_grad_norm: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int


@torch.no_grad()
def evaluate_on_windows(
    q_net: GRUQNetwork,
    store: WindowStore,
    env_cfg: EnvConfig,
    device: torch.device,
    max_eval_episodes: int,
    eval_seed: int,
) -> Dict[str, float]:
    """Evaluate greedily and restore global RNG state afterwards."""
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    q_net.eval()

    try:
        indices = list(range(len(store)))
        random.shuffle(indices)
        if max_eval_episodes > 0:
            indices = indices[:max_eval_episodes]

        returns, best_ratios, handover_rates = [], [], []
        for ep_idx in indices:
            env = RUWindowEnv(store, [ep_idx], env_cfg, training=False)
            obs = env.reset(ep_idx=ep_idx)
            hidden = q_net.init_hidden(1, device)
            ep_return = ep_best = ep_handover = 0.0
            steps = 0
            done = False
            while not done:
                obs_tensor = torch.from_numpy(obs).to(device).view(1, -1)
                q_values, hidden = q_net(obs_tensor, hidden)
                action = int(torch.argmax(q_values, dim=-1).item())
                obs, reward, done, info = env.step(action)
                ep_return += float(reward)
                ep_best += float(info["is_best_action"])
                ep_handover += float(info["handover_attempt"])
                steps += 1

            returns.append(ep_return / max(steps, 1))
            best_ratios.append(ep_best / max(steps, 1))
            handover_rates.append(ep_handover / max(steps, 1))

        return {
            "reward_mean": float(np.mean(returns)),
            "best_action_ratio": float(np.mean(best_ratios)),
            "handover_attempt_per_step": float(np.mean(handover_rates)),
            "n_episodes": int(len(returns)),
        }
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_checkpoint(
    q_net: GRUQNetwork,
    target_net: GRUQNetwork,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    env_iter: int,
    env_cfg: EnvConfig,
    dqn_cfg: DQNConfig,
    train_store: WindowStore,
    dataset_root: Path,
    args: argparse.Namespace,
    validation_metrics: Optional[Dict[str, float]],
) -> Dict:
    return {
        "global_step": int(global_step),
        "env_iter": int(env_iter),
        "q_net": cpu_state_dict(q_net),
        "target_net": cpu_state_dict(target_net),
        "optimizer": optimizer.state_dict(),
        "env_cfg": asdict(env_cfg),
        "dqn_cfg": asdict(dqn_cfg),
        "dataset_root": str(dataset_root.resolve()),
        "validation_metrics": validation_metrics,
        "args": vars(args),
        "data_meta": {
            "num_antennas": train_store.num_antennas,
            "ru_antenna_counts": list(train_store.ru_antenna_counts),
            "num_rus": train_store.num_rus,
            "action_states": [list(x) for x in train_store.action_states],
            "action_names": list(train_store.action_names),
            "dt": train_store.dt,
            "window_length": train_store.L,
            "scale": train_store.scale,
        },
        "model_meta": {
            "obs_dim": q_net.obs_dim,
            "hidden_size": q_net.hidden_size,
            "n_actions": q_net.n_actions,
            "n_params": count_trainable_params(q_net),
            "algorithm": "double_dqn_gru",
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./data_ready/dmimo_3tx_s66_s75/scheduler"),
        help="Directory containing train/, validation/, and test/",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("./checkpoints_ru_scheduler"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval-seed", type=int, default=456)
    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-env-steps", type=int, default=1_500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=2_000)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=150_000)
    parser.add_argument("--hidden-size", type=int, default=8)

    parser.add_argument("--segment-ms", type=float, default=100.0)
    parser.add_argument("--obs-history-bins", type=int, default=3)
    parser.add_argument("--mask-value", type=float, default=0.0)
    parser.add_argument("--handover-penalty", type=float, default=0.05)
    parser.add_argument("--two-ru-penalty", type=float, default=0.0)
    parser.add_argument("--reward-offset", type=float, default=3.0)
    parser.add_argument("--reward-snr-db", type=float, default=10.0)
    parser.add_argument("--noise-power", type=float, default=1e-3)
    parser.add_argument("--inter-ru-phase-noise-std", type=float, default=math.pi / 8.0)
    parser.add_argument("--max-episode-steps", type=int, default=0)

    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--validation-max-episodes", type=int, default=200)
    parser.add_argument(
        "--test-max-episodes",
        type=int,
        default=0,
        help="0 evaluates every test window",
    )
    parser.add_argument("--save-every", type=int, default=50_000)

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("[ARGS] Ignoring unknown args (likely Jupyter):", unknown)
    return args


def main(argv: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    args = parse_args(argv)
    seed_all(args.seed)
    device = get_device(args.gpu_id)
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[CUDA] {torch.cuda.get_device_name(device)}")

    stores = {
        split: WindowStore(args.dataset_root / split)
        for split in ("train", "validation", "test")
    }
    validate_split_compatibility(stores)
    train_store = stores["train"]

    segment_seconds = args.segment_ms / 1000.0
    seg_len_ticks = int(round(segment_seconds / train_store.dt))
    if seg_len_ticks <= 0 or not math.isclose(
        seg_len_ticks * train_store.dt,
        segment_seconds,
        rel_tol=0.0,
        abs_tol=max(1e-12, train_store.dt * 1e-6),
    ):
        raise ValueError(
            f"segment_ms={args.segment_ms} cannot be represented exactly with "
            f"dt={train_store.dt}; nearest ticks={seg_len_ticks}"
        )

    print(f"[DATA] root={args.dataset_root.resolve()}")
    for split, store in stores.items():
        print(
            f"[DATA] {split:10s} X={store.X.shape} "
            f"scenarios={list(store.scenario_ids)}"
        )
    print(
        f"[LAYOUT] antennas={train_store.num_antennas} "
        f"ru_antenna_counts={list(train_store.ru_antenna_counts)} "
        f"actions={list(train_store.action_names)}"
    )

    env_cfg = EnvConfig(
        seg_len_ticks=seg_len_ticks,
        obs_hist_bins=args.obs_history_bins,
        mask_value=args.mask_value,
        handover_penalty=args.handover_penalty,
        inter_ru_phase_noise_std=args.inter_ru_phase_noise_std,
        two_ru_penalty=args.two_ru_penalty,
        noise_power=args.noise_power,
        reward_snr_db=args.reward_snr_db,
        reward_offset=args.reward_offset,
        max_episode_steps=args.max_episode_steps,
    )
    dqn_cfg = DQNConfig(
        total_env_steps=args.total_env_steps,
        num_envs=args.num_envs,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        learning_starts=args.learning_starts,
        train_every=args.train_every,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        max_grad_norm=args.max_grad_norm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )

    train_indices = list(range(len(train_store)))
    train_envs = [
        RUWindowEnv(train_store, train_indices, env_cfg, training=True)
        for _ in range(dqn_cfg.num_envs)
    ]
    vector_env = VecEnv(train_envs)
    obs = vector_env.reset()

    q_net = GRUQNetwork(vector_env.obs_dim, args.hidden_size, vector_env.n_actions).to(device)
    target_net = GRUQNetwork(vector_env.obs_dim, args.hidden_size, vector_env.n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    for parameter in target_net.parameters():
        parameter.requires_grad_(False)

    print(
        f"[MODEL] obs_dim={q_net.obs_dim} hidden={q_net.hidden_size} "
        f"actions={q_net.n_actions} parameters={count_trainable_params(q_net)}"
    )
    optimizer = torch.optim.Adam(q_net.parameters(), lr=dqn_cfg.lr, eps=1e-5)
    replay = ReplayBuffer(dqn_cfg.replay_size, vector_env.obs_dim, q_net.hidden_size)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    hidden_env = q_net.init_hidden(dqn_cfg.num_envs, device).cpu().numpy()
    running_return = np.zeros(dqn_cfg.num_envs, dtype=np.float64)
    running_best = np.zeros(dqn_cfg.num_envs, dtype=np.float64)
    running_handover = np.zeros(dqn_cfg.num_envs, dtype=np.float64)
    running_length = np.zeros(dqn_cfg.num_envs, dtype=np.int64)
    recent_returns = deque(maxlen=100)
    recent_best = deque(maxlen=100)
    recent_handover = deque(maxlen=100)
    recent_losses = deque(maxlen=200)

    global_step = env_iter = last_target_sync = 0
    next_log_step = args.log_every
    next_eval_step = args.eval_every
    next_save_step = args.save_every
    started = time.time()
    best_validation_reward = -math.inf
    best_validation_step = 0
    best_validation_metrics: Optional[Dict[str, float]] = None
    best_q_state: Optional[Dict[str, torch.Tensor]] = None

    while global_step < dqn_cfg.total_env_steps:
        q_net.train()
        epsilon = linear_schedule(
            global_step,
            dqn_cfg.epsilon_start,
            dqn_cfg.epsilon_end,
            dqn_cfg.epsilon_decay_steps,
        )
        obs_tensor = torch.from_numpy(obs).to(device)
        hidden_tensor = torch.from_numpy(hidden_env).to(device)
        with torch.no_grad():
            q_values, next_hidden_tensor = q_net(obs_tensor, hidden_tensor)
            greedy_actions = q_values.argmax(dim=-1).cpu().numpy()

        random_actions = np.random.randint(
            0, vector_env.n_actions, size=dqn_cfg.num_envs, dtype=np.int64
        )
        explore = np.random.rand(dqn_cfg.num_envs) < epsilon
        actions = np.where(explore, random_actions, greedy_actions).astype(np.int64)
        next_obs, rewards, dones, infos = vector_env.step(actions)

        next_hidden = next_hidden_tensor.cpu().numpy().astype(np.float32)
        for i, info in enumerate(infos):
            if info["episode_reset"]:
                next_hidden[i] = 0.0
            running_return[i] += float(rewards[i])
            running_best[i] += float(info["is_best_action"])
            running_handover[i] += float(info["handover_attempt"])
            running_length[i] += 1
            if dones[i]:
                length = max(int(running_length[i]), 1)
                recent_returns.append(float(running_return[i] / length))
                recent_best.append(float(running_best[i] / length))
                recent_handover.append(float(running_handover[i] / length))
                running_return[i] = running_best[i] = running_handover[i] = 0.0
                running_length[i] = 0

        replay.add_batch(
            obs, hidden_env, actions, rewards, next_obs, next_hidden, dones.astype(np.float32)
        )
        obs = next_obs
        hidden_env = next_hidden
        env_iter += 1
        global_step += dqn_cfg.num_envs

        if (
            global_step >= dqn_cfg.learning_starts
            and len(replay) >= dqn_cfg.batch_size
            and env_iter % dqn_cfg.train_every == 0
        ):
            for _ in range(dqn_cfg.gradient_steps):
                (
                    batch_obs,
                    batch_hidden,
                    batch_actions,
                    batch_rewards,
                    batch_next_obs,
                    batch_next_hidden,
                    batch_dones,
                ) = replay.sample(dqn_cfg.batch_size, device)
                q_all, _ = q_net(batch_obs, batch_hidden)
                q_pred = q_all.gather(1, batch_actions[:, None]).squeeze(1)
                with torch.no_grad():
                    online_next_q, _ = q_net(batch_next_obs, batch_next_hidden)
                    next_actions = online_next_q.argmax(dim=-1, keepdim=True)
                    target_next_q, _ = target_net(batch_next_obs, batch_next_hidden)
                    next_q = target_next_q.gather(1, next_actions).squeeze(1)
                    target = (
                        batch_rewards
                        + dqn_cfg.gamma * (1.0 - batch_dones) * next_q
                    )
                loss = F.smooth_l1_loss(q_pred, target)
                recent_losses.append(float(loss.item()))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), dqn_cfg.max_grad_norm)
                optimizer.step()

        if global_step - last_target_sync >= dqn_cfg.target_update_interval:
            target_net.load_state_dict(q_net.state_dict())
            last_target_sync = global_step

        if args.log_every > 0 and global_step >= next_log_step:
            elapsed_min = (time.time() - started) / 60.0
            mean = lambda values: float(np.mean(values)) if values else float("nan")
            print(
                f"[TRAIN step={global_step:9d}] eps={epsilon:.3f} "
                f"buffer={len(replay):6d} loss={mean(recent_losses):.4f} "
                f"reward/step={mean(recent_returns):.4f} "
                f"best_ratio={mean(recent_best):.4f} "
                f"handover/step={mean(recent_handover):.4f} "
                f"elapsed={elapsed_min:.1f}min"
            )
            next_log_step += args.log_every

        if args.eval_every > 0 and global_step >= next_eval_step:
            metrics = evaluate_on_windows(
                q_net,
                stores["validation"],
                env_cfg,
                device,
                args.validation_max_episodes,
                args.eval_seed,
            )
            print(
                f"[VALIDATION step={global_step:9d}] "
                f"reward/step={metrics['reward_mean']:.4f} "
                f"best_ratio={metrics['best_action_ratio']:.4f} "
                f"handover/step={metrics['handover_attempt_per_step']:.4f} "
                f"episodes={metrics['n_episodes']}"
            )
            if metrics["reward_mean"] > best_validation_reward:
                best_validation_reward = metrics["reward_mean"]
                best_validation_step = global_step
                best_validation_metrics = metrics
                best_q_state = cpu_state_dict(q_net)
                checkpoint = build_checkpoint(
                    q_net,
                    target_net,
                    optimizer,
                    global_step,
                    env_iter,
                    env_cfg,
                    dqn_cfg,
                    train_store,
                    args.dataset_root,
                    args,
                    metrics,
                )
                torch.save(checkpoint, args.save_dir / "best_validation.pt")
                print(f"[BEST] {args.save_dir / 'best_validation.pt'}")
            next_eval_step += args.eval_every

        if args.save_every > 0 and global_step >= next_save_step:
            checkpoint = build_checkpoint(
                q_net,
                target_net,
                optimizer,
                global_step,
                env_iter,
                env_cfg,
                dqn_cfg,
                train_store,
                args.dataset_root,
                args,
                best_validation_metrics,
            )
            path = args.save_dir / f"scheduler_step_{global_step:09d}.pt"
            torch.save(checkpoint, path)
            print(f"[SAVE] {path}")
            next_save_step += args.save_every

    if best_q_state is None:
        metrics = evaluate_on_windows(
            q_net,
            stores["validation"],
            env_cfg,
            device,
            args.validation_max_episodes,
            args.eval_seed,
        )
        best_validation_reward = metrics["reward_mean"]
        best_validation_step = global_step
        best_validation_metrics = metrics
        best_q_state = cpu_state_dict(q_net)
        checkpoint = build_checkpoint(
            q_net,
            target_net,
            optimizer,
            global_step,
            env_iter,
            env_cfg,
            dqn_cfg,
            train_store,
            args.dataset_root,
            args,
            metrics,
        )
        torch.save(checkpoint, args.save_dir / "best_validation.pt")

    # Test is deliberately evaluated once, after all training/model selection.
    q_net.load_state_dict(best_q_state)
    test_metrics = evaluate_on_windows(
        q_net,
        stores["test"],
        env_cfg,
        device,
        args.test_max_episodes,
        args.eval_seed + 1,
    )
    results = {
        "best_validation_step": int(best_validation_step),
        "validation": best_validation_metrics,
        "test": test_metrics,
    }
    with (args.save_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    final_checkpoint = build_checkpoint(
        q_net,
        target_net,
        optimizer,
        best_validation_step,
        env_iter,
        env_cfg,
        dqn_cfg,
        train_store,
        args.dataset_root,
        args,
        best_validation_metrics,
    )
    final_checkpoint["test_metrics"] = test_metrics
    torch.save(final_checkpoint, args.save_dir / "final_best_model.pt")

    print(f"[DONE] best validation step={best_validation_step}")
    print(f"[TEST] {json.dumps(test_metrics, ensure_ascii=False)}")
    print(f"[RESULTS] {args.save_dir.resolve()}")
    return results


if __name__ == "__main__":
    main()
