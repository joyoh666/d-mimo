#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset -> channel predictor -> RU scheduler -> dashboard.

The dashboard layout follows the supplied ``live_dashboard.py``, but this
version reads a finite dataset directly instead of waiting for UDP packets.

Supported inputs
----------------
* A receiver run directory containing ``usrp_sparse_h.npz`` (``csi`` key).
* ``.npz`` with ``csi`` shaped ``(T, 3)`` (raw CSI).
* ``.npz`` with ``X`` shaped ``(N, T, 6)`` (pre-scaled training windows).
* ``.npy`` shaped ``(T, 3)``, ``(T, 6)``, or ``(N, T, C)``.
* ``.csv`` with three complex columns, or real/imaginary column pairs.

Both exported NumPy models (``.npz``) and the PyTorch checkpoints written by
``train_GRU.py`` / the RU scheduler trainer (``.pt`` or ``.pth``) are accepted.

Examples
--------
Preview with generated CSI::

    .venv/bin/python dataset_dashboard.py \
        --predictor-model models/predictor.npz \
        --dqn-model models/dqn.npz

Run a measured receiver dataset::

    .venv/bin/python dataset_dashboard.py --data path/to/run_directory \
        --scale data_ready/scale.json

Run one window from the training dataset::

    .venv/bin/python dataset_dashboard.py \
        --data data_ready/pred_ws6p00s/X.npz --sample-index 0

Controls: Space = play/pause, Left/Right = seek 1 s, Home/End = start/end,
R = restart, Q/Escape = close.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hj_common as hc  # noqa: E402


RU_COLORS = ["#4FC3F7", "#FFB74D", "#81C784"]
TARGET_POWER = 1e-2


@dataclass
class LoadedDataset:
    name: str
    values: np.ndarray
    input_kind: str
    dt: float
    source: Path | None
    selected_key: str


def _synthetic_csi(seconds: float = 15.0, seed: int = 7) -> LoadedDataset:
    """Smooth three-RU CSI used only to verify that the complete UI runs."""
    rng = np.random.default_rng(seed)
    n = max(200, int(round(seconds / hc.DT)))
    t = np.arange(n, dtype=np.float64) * hc.DT
    out = np.empty((n, 3), np.complex64)
    for ru in range(3):
        slow = 1.0 + 0.28 * np.sin(2 * np.pi * (0.10 + 0.025 * ru) * t + 1.1 * ru)
        fast = 0.10 * np.sin(2 * np.pi * (0.7 + 0.12 * ru) * t + 0.4 * ru)
        amp = (0.0035 + 0.0012 * ru) * np.maximum(0.15, slow + fast)
        phase = (
            2 * np.pi * (0.16 + 0.035 * ru) * t
            + 0.22 * np.sin(2 * np.pi * 0.23 * t + ru)
            + np.cumsum(rng.normal(0.0, 0.0025, size=n))
        )
        noise = 8e-5 * (rng.normal(size=n) + 1j * rng.normal(size=n))
        out[:, ru] = amp * np.exp(1j * phase) + noise
    return LoadedDataset("synthetic CSI", out, "raw-csi", hc.DT, None, "synthetic")


def _read_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        names = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")

    lower = {name.lower(): name for name in names}
    pairs: list[tuple[str, str]] = []
    for i in range(1, 4):
        found = None
        for stem in (f"ru{i}", f"ru_{i}", f"h{i}", f"h_{i}", f"csi{i}", f"csi_{i}", f"tx{i}", f"tx_{i}"):
            for re_suffix, im_suffix in (("_real", "_imag"), ("_re", "_im"), (".real", ".imag")):
                if stem + re_suffix in lower and stem + im_suffix in lower:
                    found = (lower[stem + re_suffix], lower[stem + im_suffix])
                    break
            if found:
                break
        if found:
            pairs.append(found)
    if len(pairs) == 3:
        return np.asarray(
            [[float(row[re_name]) + 1j * float(row[im_name]) for re_name, im_name in pairs] for row in rows],
            dtype=np.complex64,
        )

    ignored = {"t", "time", "timestamp", "t_os", "t_usrp", "frame", "frame_idx", "sym", "sym_in_frame"}
    complex_columns: list[str] = []
    for name in names:
        if name.lower() in ignored:
            continue
        try:
            complex(rows[0][name].strip().strip("() ").replace("i", "j"))
        except (TypeError, ValueError):
            continue
        complex_columns.append(name)
    if len(complex_columns) < 3:
        raise ValueError(
            "CSV must contain ru1_real/ru1_imag ... ru3_real/ru3_imag, "
            "or at least three columns containing complex numbers."
        )
    cols = complex_columns[:6] if len(complex_columns) >= 6 else complex_columns[:3]
    return np.asarray(
        [[complex(row[name].strip().strip("() ").replace("i", "j")) for name in cols] for row in rows],
        dtype=np.complex64,
    )


def _choose_npz_key(z: Any, requested: str | None) -> tuple[str, str]:
    if requested:
        if requested not in z.files:
            raise KeyError(f"NPZ key {requested!r} not found; available keys: {z.files}")
        kind = "raw-csi" if requested.lower() in {"csi", "raw", "channels"} else "model-x"
        return requested, kind
    for key, kind in (("csi", "raw-csi"), ("X", "model-x"), ("x", "model-x"), ("channels", "raw-csi"), ("h", "raw-csi")):
        if key in z.files:
            return key, kind
    array_keys = [key for key in z.files if np.asarray(z[key]).ndim in (2, 3)]
    if len(array_keys) == 1:
        return array_keys[0], "raw-csi"
    raise KeyError(f"Could not choose the CSI array; use --key. Available keys: {z.files}")


def _slice_sample(values: np.ndarray, sample_index: int) -> np.ndarray:
    if values.ndim == 3:
        if not -values.shape[0] <= sample_index < values.shape[0]:
            raise IndexError(f"sample index {sample_index} is outside 0..{values.shape[0] - 1}")
        values = values[sample_index]
    if values.ndim != 2:
        raise ValueError(f"CSI must have shape (T,C) or (N,T,C), got {values.shape}")
    if values.shape[1] not in (3, 6):
        raise ValueError(f"Expected 3 raw CSI columns or 6 model-X columns, got {values.shape}")
    if not np.iscomplexobj(values):
        raise ValueError("The selected array is real-valued; provide complex CSI or use a CSV with real/imaginary pairs.")
    values = np.asarray(values, dtype=np.complex64)
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("CSI contains NaN or infinite values.")
    return values


def load_dataset(path_text: str | None, key: str | None, sample_index: int, input_kind: str) -> LoadedDataset:
    if not path_text:
        return _synthetic_csi()

    path = Path(path_text).expanduser().resolve()
    run_dir: Path | None = None
    if path.is_dir():
        run_dir = path
        path = path / "usrp_sparse_h.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    detected_kind = "raw-csi"
    selected_key = key or ""
    dt = hc.DT
    time_values: np.ndarray | None = None
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            selected_key, detected_kind = _choose_npz_key(z, key)
            values = np.asarray(z[selected_key])
            if "dt" in z.files:
                dt = float(np.asarray(z["dt"]).item())
            for time_key in ("csi_t_usrp", "t", "time"):
                if time_key in z.files and np.asarray(z[time_key]).ndim == 1:
                    time_values = np.asarray(z[time_key], dtype=np.float64)
                    break
    elif path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        selected_key = "array"
        detected_kind = "model-x" if values.shape[-1] == 6 else "raw-csi"
    elif path.suffix.lower() == ".csv":
        values = _read_csv(path)
        selected_key = "csv columns"
        detected_kind = "model-x" if values.shape[-1] == 6 else "raw-csi"
    else:
        raise ValueError("Supported dataset types are a receiver run directory, .npz, .npy, and .csv")

    # A receiver run can contain CSI outside the scenario window. Match build_dataset.py.
    marker = (run_dir / "scenario_marker.json") if run_dir else path.with_name("scenario_marker.json")
    if marker.exists() and time_values is not None and values.ndim == 2 and len(time_values) == len(values):
        info = json.loads(marker.read_text(encoding="utf-8"))
        if "t0_usrp_est" in info and "t1_usrp_est" in info:
            selected = (time_values >= float(info["t0_usrp_est"])) & (time_values < float(info["t1_usrp_est"]))
            values = values[selected]
            time_values = time_values[selected]
    if time_values is not None and len(time_values) > 1 and values.ndim == 2:
        measured_dt = float(np.median(np.diff(time_values)))
        if measured_dt > 0:
            dt = measured_dt

    values = _slice_sample(np.asarray(values), sample_index)
    kind = detected_kind if input_kind == "auto" else input_kind
    if abs(dt - hc.DT) > 5e-4:
        raise ValueError(
            f"Dataset interval is {dt:.6f} s, but the models require {hc.DT:.3f} s CSI. "
            "Resample the dataset to 5 ms first."
        )
    return LoadedDataset(path.name, values, kind, dt, path, selected_key)


def _read_scale(spec: str, values: np.ndarray, dataset: LoadedDataset) -> tuple[float, str, bool]:
    if dataset.input_kind == "model-x":
        return 1.0, "pre-scaled X", values.shape[1] == 6
    if spec.lower() == "auto":
        power = float(np.mean(np.abs(values) ** 2))
        if power <= 0:
            raise ValueError("Cannot auto-scale an all-zero dataset.")
        return math.sqrt(TARGET_POWER / power), "auto", False
    candidate = Path(spec).expanduser()
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        legacy_hint = "to_x6" in str(payload.get("note", "")).lower()
        return float(payload["scale"]), str(candidate.resolve()), legacy_hint
    try:
        return float(spec), "command line", False
    except ValueError as exc:
        raise ValueError("--scale must be 'auto', a number, or a scale.json path") from exc


def prepare_model_input(
    dataset: LoadedDataset,
    scale_spec: str,
    representation: str,
) -> tuple[np.ndarray, float, str, bool]:
    values = dataset.values
    scale, scale_source, legacy_hint = _read_scale(scale_spec, values, dataset)
    duplicate_rus = legacy_hint if representation == "auto" else representation == "legacy-x6"
    if dataset.input_kind == "raw-csi":
        if values.shape[1] != 3:
            raise ValueError("raw-csi input must have shape (T,3); use --input-kind model-x for six-column X")
        # Legacy training duplicated h/sqrt(2) into two antenna columns per RU.
        # The current pipeline preserves the three input columns and scales h directly.
        divisor = math.sqrt(2.0) if duplicate_rus else 1.0
        x = values * np.float32(scale / divisor)
    else:
        if values.shape[1] == 6:
            if representation == "preserve":
                raise ValueError("Six-column model-X input requires --representation auto or legacy-x6")
            duplicate_rus = True
            pair_error = float(np.mean(np.abs(values[:, 0::2] - values[:, 1::2])))
            pair_level = float(np.mean(np.abs(values))) + 1e-12
            if pair_error / pair_level > 0.05:
                raise ValueError(
                    "Six-column model-X contains distinct antenna channels. This three-plot dashboard currently "
                    "supports three channels or legacy duplicated RU pairs, not collapsing distinct antennas."
                )
            x = values[:, 0::2]
        else:
            if representation == "legacy-x6":
                duplicate_rus = True
            elif representation in {"auto", "preserve"}:
                duplicate_rus = False
            x = values
    return np.asarray(x, dtype=np.complex64), scale, scale_source, duplicate_rus


def resolve_model(path_text: str, model_dir: str | None, fallback_patterns: tuple[str, ...] = ()) -> Path:
    requested = Path(path_text).expanduser()
    candidates = [requested]
    if model_dir:
        candidates.insert(0, Path(model_dir).expanduser() / requested.name)
    if not requested.is_absolute():
        candidates.extend([HERE / requested, HERE / "models" / requested.name])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    for pattern in fallback_patterns:
        matches = sorted(HERE.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        if matches:
            selected = matches[0].resolve()
            print(f"[MODEL] {requested.name} not found; using checkpoint {selected}")
            return selected
    checked = "\n  - ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"Model {requested.name!r} was not found. Checked:\n  - {checked}\nUse --model-dir or an explicit model path.")


class OfflinePipeline:
    """Runs the same predictor and scheduling equations as live_dashboard.py."""

    def __init__(
        self,
        x: np.ndarray,
        predictor: hc.NumpyPredictor,
        dqn: hc.NumpyDQN,
        args: argparse.Namespace,
        duplicate_rus: bool,
    ):
        self.x = x
        self.args = args
        self.pred = predictor
        self.dqn = dqn
        self.taps = self.pred.taps
        self.K = self.pred.K
        self.seg = int(self.dqn.cfg["seg_len_ticks"])
        self.hist_bins = int(self.dqn.cfg["obs_hist_bins"])
        self.phase_std = float(self.dqn.cfg["inter_ru_phase_noise_std"])
        self.noise_power = float(self.dqn.cfg["noise_power"])
        self.snr_db = float(self.dqn.cfg.get("reward_snr_db", 10.0))
        self.pred_at = np.full((self.K, len(x), 3), np.nan + 0j, np.complex64)
        self.forecasts: dict[int, np.ndarray] = {}
        self.err_hist: list[tuple[int, int, float, float, float, float]] = []
        self.ru_gain: list[np.ndarray] = []
        self.state_rate: list[np.ndarray] = []
        self.serve = (
            [args.init_state]
            if self.dqn.timing == "next-segment"
            else [args.init_state, args.init_state]
        )
        self.q_hist: list[np.ndarray] = []
        self.rng = np.random.RandomState(args.seed)
        self.duplicate_rus = duplicate_rus

    def run(self) -> None:
        total_steps = len(self.x) // self.taps
        print(f"[RUN ] channel predictor: {total_steps} x {self.taps * hc.DT * 1000:.0f} ms steps")
        for s in range(total_steps):
            self._predict_step(s)
            if total_steps >= 20 and (s + 1) % max(1, total_steps // 10) == 0:
                print(f"       {100 * (s + 1) / total_steps:5.1f}%")
        total_segments = len(self.x) // self.seg
        print(f"[RUN ] RU scheduler: {total_segments} x {self.seg * hc.DT * 1000:.0f} ms segments")
        for c in range(total_segments):
            self._schedule_segment(c)

    def _predict_step(self, s: int) -> None:
        tp, K = self.taps, self.K
        end = (s + 1) * tp
        for k in range(K):
            src = s - (k + 1)
            if src < 0:
                continue
            pred = self.pred_at[k, s * tp:end]
            if np.isnan(pred.real).any():
                continue
            gt = self.x[s * tp:end]
            lr_err, kf_err = self._baseline_err(src, k, gt)
            self.err_hist.append(
                (end, k, float(np.mean(np.abs(pred - gt) ** 2)), float(np.mean(np.abs(gt) ** 2)), lr_err, kf_err)
            )
        context = min(self.args.context_steps, s + 1)
        if context < self.args.min_context_steps:
            return
        history = self.x[end - context * tp:end].T
        future = self.pred.predict_complex(history).reshape(3, K, tp)
        for k in range(K):
            a, b = end + k * tp, end + (k + 1) * tp
            if b <= len(self.x):
                self.pred_at[k, a:b] = future[:, k].T
        self.forecasts[end] = future.reshape(3, -1)

    def _baseline_err(self, src: int, k: int, gt: np.ndarray) -> tuple[float, float]:
        tp = self.taps
        gt0 = self.x[src * tp:(src + 1) * tp].T
        lr = hc.lr_predict(gt0, (k + 1) * tp)[:, k * tp:].T
        kf = hc.kf_predict(gt0, (k + 1) * tp)[:, k * tp:].T
        return float(np.mean(np.abs(lr - gt) ** 2)), float(np.mean(np.abs(kf - gt) ** 2))

    def _schedule_segment(self, c: int) -> None:
        a, b = c * self.seg, (c + 1) * self.seg
        gains = np.abs(self.x[a:b])
        if self.duplicate_rus:
            gains = np.repeat(gains, 2, axis=1)
        self.ru_gain.append(hc.ru_gain_segment(gains))
        self.state_rate.append(
            hc.all_state_power_segment(gains, self.phase_std, self.noise_power, self.snr_db, self.rng)
        )
        if self.dqn.timing == "next-segment":
            # train_RU_scheduler.py: observe history through segment c-1 and
            # choose the connection used by reward/service segment c+1.
            seg_k = c
            connection_state = self.serve[c]
        else:
            # Legacy scheduler: the action selected after segment c is served
            # by segment c+2.
            seg_k = c + 1
            connection_state = self.serve[seg_k]
        obs = hc.make_obs(self.ru_gain, self.serve, seg_k, connection_state, self.hist_bins)
        action, q = self.dqn.step(obs)
        self.serve.append(action)
        self.q_hist.append(q)

    def nmse_db(self, n: int, last_ticks: int) -> dict[str, list[float]]:
        lo = max(0, n - last_ticks)
        rows = [row for row in self.err_hist if lo < row[0] <= n]
        out: dict[str, list[float]] = {}
        for name, error_index in (("MODEL", 2), ("LR", 4), ("KF", 5)):
            values = []
            for k in range(self.K):
                selected = [row for row in rows if row[1] == k]
                numerator = sum(row[error_index] for row in selected)
                denominator = sum(row[3] for row in selected)
                values.append(10 * np.log10(numerator / denominator) if denominator > 0 and numerator > 0 else np.nan)
            out[name] = values
        return out

    def rate_summary(self, n: int, last_segments: int) -> dict[str, float]:
        count = min(len(self.state_rate), n // self.seg)
        lo = max(0, count - last_segments)
        if count <= lo:
            return {}
        rates = np.asarray(self.state_rate[lo:count])
        selected = np.asarray(self.serve[lo:count], dtype=np.int64)
        out = {
            "DQN": float(rates[np.arange(len(selected)), selected].mean()),
            "Oracle": float(rates.max(axis=1).mean()),
            "Random": float(rates.mean()),
        }
        for index, name in enumerate(hc.STATE_NAMES):
            out[name] = float(rates[:, index].mean())
        out["best_ratio"] = float(np.mean(selected == rates.argmax(axis=1)))
        out["handover"] = float(np.mean(np.diff(np.asarray(self.serve[lo:count + 1])) != 0)) if count - lo > 1 else 0.0
        return out

    def forecast_at(self, n: int) -> tuple[int, np.ndarray] | None:
        end = (n // self.taps) * self.taps
        while end >= 0:
            if end in self.forecasts:
                return end, self.forecasts[end]
            end -= self.taps
        return None


def save_results(path: Path, pipe: OfflinePipeline, dataset: LoadedDataset, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=pipe.x,
        pred_at=pipe.pred_at,
        ru_gain=np.asarray(pipe.ru_gain),
        state_rate=np.asarray(pipe.state_rate),
        serve=np.asarray(pipe.serve),
        q=np.asarray(pipe.q_hist),
        dt=np.float32(hc.DT),
        scale=np.float64(scale),
        representation=np.str_("legacy-x6" if pipe.duplicate_rus else "preserve"),
        source=np.str_(str(dataset.source or "synthetic")),
    )
    print(f"[SAVE] {path.resolve()}")


def _configure_matplotlib(snapshot: Path | None):
    import matplotlib

    if snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(["dark_background", "fast"])
    have = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    korean = next(
        (
            name
            for name in (
                "AppleGothic",
                "Malgun Gothic",
                "NanumGothic",
                "NanumBarunGothic",
                "Noto Sans CJK KR",
                "Noto Sans CJK JP",
                "Noto Sans KR",
            )
            if name in have
        ),
        None,
    )
    if korean:
        plt.rcParams["font.family"] = [korean, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def run_gui(
    pipe: OfflinePipeline,
    dataset: LoadedDataset,
    scale: float,
    scale_source: str,
    args: argparse.Namespace,
    snapshot: Path | None,
) -> None:
    plt = _configure_matplotlib(snapshot)
    from matplotlib.animation import FuncAnimation

    window = args.window_sec
    tp, K, seg = pipe.taps, pipe.K, pipe.seg
    fig = plt.figure(figsize=(16, 9))
    try:
        fig.canvas.manager.set_window_title("D-MIMO dataset dashboard")
    except AttributeError:
        pass
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.35, 1],
        hspace=0.38,
        wspace=0.18,
        left=0.05,
        right=0.98,
        top=0.80,
        bottom=0.07,
    )
    title = fig.suptitle("", fontsize=13, y=0.985)
    box = dict(boxstyle="round,pad=0.45", fc="#1E1E1E", ec="0.45")
    acc_predict = fig.text(0.05, 0.91, "", fontsize=12, va="center", ha="left", bbox=box, linespacing=1.4)
    acc_schedule = fig.text(0.585, 0.91, "", fontsize=12, va="center", ha="left", bbox=box, linespacing=1.4)
    fig.text(
        0.05,
        0.858,
        "정확도 = 1 − NMSE (예측 오차 에너지 ÷ 실제 채널 에너지).  LR = 직선 외삽 비교 기준",
        fontsize=8.5,
        color="0.65",
        va="top",
    )
    fig.text(
        0.585,
        0.858,
        "최선 RU 맞힘 = 다음 100 ms에서 rate가 가장 높은 조합 선택 비율 · 오라클 = 정답을 아는 선택",
        fontsize=8.5,
        color="0.65",
        va="top",
    )

    def accuracy(db_value: float) -> float:
        return np.nan if not np.isfinite(db_value) else max(0.0, 1.0 - 10 ** (db_value / 10.0)) * 100.0

    left_axes = []
    left_lines = []
    for ru in range(3):
        ax = fig.add_subplot(grid[ru, 0])
        ground_truth, = ax.plot([], [], color=RU_COLORS[ru], lw=1.6, label="실측 |h|")
        pred_1, = ax.plot([], [], color="w", lw=0.9, ls=":", alpha=0.8, label="GRU 예측 +25 ms")
        pred_5, = ax.plot([], [], color="#FF5252", lw=1.1, ls="--", alpha=0.9, label="GRU 예측 +125 ms")
        forecast, = ax.plot([], [], color="#FF5252", lw=2.4, label="지금부터 125 ms 예측")
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlim(-window, K * tp * hc.DT + 0.05)
        ax.set_ylabel(f"RU{ru + 1}  |h| [dB]")
        ax.grid(alpha=0.25)
        if ru == 0:
            ax.set_title("채널 예측 (GRU, 25 ms × 5 스텝)", loc="left", fontsize=11)
            ax.legend(loc="lower left", fontsize=8, ncol=4)
        if ru == 2:
            ax.set_xlabel("시간 [s] (0 = 현재 재생 위치)")
        text = ax.text(0.995, 0.96, "", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, family="monospace")
        left_axes.append(ax)
        left_lines.append((ground_truth, pred_1, pred_5, forecast, text))

    gain_ax = fig.add_subplot(grid[0, 1])
    gain_lines = [gain_ax.step([], [], where="post", color=RU_COLORS[ru], lw=1.6, label=f"RU{ru + 1}")[0] for ru in range(3)]
    gain_ax.set_title("RU 이득 (100 ms 세그먼트)", loc="left", fontsize=11)
    gain_ax.set_ylabel("RU gain")
    gain_ax.legend(loc="upper left", fontsize=8, ncol=3)
    gain_ax.grid(alpha=0.25)
    gain_ax.set_xlim(-window, 0.2)

    action_ax = fig.add_subplot(grid[1, 1], sharex=gain_ax)
    action_line, = action_ax.step([], [], where="post", color="#FFEB3B", lw=2.2, label="DQN 선택")
    oracle_points = action_ax.scatter([], [], s=14, color="#E040FB", label="오라클", zorder=3)
    action_ax.set_yticks(range(6))
    action_ax.set_yticklabels(hc.STATE_NAMES)
    action_ax.set_ylim(-0.5, 5.5)
    action_ax.set_title("RU 스케줄링 (Double-DQN GRU)", loc="left", fontsize=11)
    action_ax.legend(loc="upper left", fontsize=8, ncol=2)
    action_ax.grid(alpha=0.25)
    action_ax.set_xlabel("시간 [s] (0 = 현재 재생 위치)")

    rate_ax = fig.add_subplot(grid[2, 1])
    rate_names = ["DQN", "Oracle", "Random"] + hc.STATE_NAMES
    bars = rate_ax.bar(
        range(len(rate_names)),
        np.zeros(len(rate_names)),
        color=["#FFEB3B", "#E040FB", "0.5"] + ["0.35"] * 6,
    )
    rate_ax.set_xticks(range(len(rate_names)))
    rate_ax.set_xticklabels(rate_names, fontsize=9)
    rate_ax.set_ylabel("평균 rate [b/s/Hz]")
    rate_ax.grid(alpha=0.25, axis="y")
    rate_title = rate_ax.set_title("", loc="left", fontsize=11)

    minimum_ticks = max(args.min_context_steps * tp, seg)
    state = {
        "n": len(pipe.x) if snapshot else min(len(pipe.x), max(minimum_ticks, int(window / hc.DT))),
        "playing": not snapshot,
    }

    def db(values: np.ndarray) -> np.ndarray:
        display_factor = math.sqrt(2.0) if pipe.duplicate_rus else 1.0
        return 20 * np.log10(np.maximum(np.abs(values), 1e-6) * display_factor)

    def update(_frame=None):
        n = int(state["n"])
        if state["playing"] and not snapshot:
            advance = max(1, int(round(args.speed / (args.fps * hc.DT))))
            n = min(len(pipe.x), n + advance)
            state["n"] = n
            if n >= len(pipe.x):
                state["playing"] = False

        window_ticks = int(window / hc.DT)
        start = max(0, n - window_ticks)
        times = (np.arange(start, n) - n) * hc.DT
        x = pipe.x[start:n]
        stats_ticks = max(1, int(args.stat_sec / hc.DT))
        nmse = pipe.nmse_db(n, stats_ticks)
        gru_1 = accuracy(nmse["MODEL"][0])
        gru_5 = accuracy(nmse["MODEL"][K - 1])
        lr_5 = accuracy(nmse["LR"][K - 1])
        if np.isfinite(gru_5):
            acc_predict.set_text(
                f"채널 예측 정확도 (최근 {args.stat_sec:.0f} s)\n"
                f"125 ms 앞: GRU {gru_5:.1f} % vs LR {lr_5:.1f} %     |     25 ms 앞: GRU {gru_1:.1f} %"
            )
        else:
            acc_predict.set_text("채널 예측 정확도\n평가 가능한 미래 샘플을 재생하는 중 …")

        current_forecast = pipe.forecast_at(n)
        for ru in range(3):
            ground_truth, pred_1, pred_5, forecast, text = left_lines[ru]
            ground_truth.set_data(times, db(x[:, ru]))
            pred_1.set_data(times, db(pipe.pred_at[0, start:n, ru]))
            pred_5.set_data(times, db(pipe.pred_at[K - 1, start:n, ru]))
            if current_forecast is not None:
                forecast_start, values = current_forecast
                forecast_times = (np.arange(forecast_start, forecast_start + values.shape[1]) - n) * hc.DT
                forecast.set_data(forecast_times, db(values[ru]))
            else:
                forecast.set_data([], [])
            y_values = db(x[:, ru])
            finite_y = y_values[np.isfinite(y_values)]
            if len(finite_y):
                low, high = np.percentile(finite_y, [1, 99])
                if high - low < 1:
                    low, high = low - 0.5, high + 0.5
                left_axes[ru].set_ylim(low - 4, high + 4)
            if ru == 0:
                text.set_text(
                    f"NMSE[dB] {args.stat_sec:.0f}s  +25ms +125ms\n"
                    + "\n".join(
                        f"{label:12s}{nmse[name][0]:+6.1f} {nmse[name][K - 1]:+6.1f}"
                        for name, label in (("MODEL", "GRU"), ("LR", "LR"), ("KF", "KF"))
                    )
                )

        segment_count = min(len(pipe.state_rate), n // seg)
        if segment_count:
            visible_segments = int(window / (seg * hc.DT)) + 1
            low_segment = max(0, segment_count - visible_segments)
            segment_times = (np.arange(low_segment, segment_count + 1) * seg - n) * hc.DT
            gains = np.asarray(pipe.ru_gain[low_segment:segment_count])
            for ru in range(3):
                gain_lines[ru].set_data(segment_times, np.r_[gains[:, ru], gains[-1, ru]])
            gain_ax.set_ylim(0, max(1e-3, float(gains.max()) * 1.15))

            served = np.asarray(pipe.serve[low_segment:segment_count + 2])
            served_times = (np.arange(low_segment, low_segment + len(served) + 1) * seg - n) * hc.DT
            action_line.set_data(served_times, np.r_[served, served[-1]])
            rates = np.asarray(pipe.state_rate[low_segment:segment_count])
            oracle_points.set_offsets(
                np.c_[
                    (np.arange(low_segment, segment_count) * seg + seg / 2 - n) * hc.DT,
                    rates.argmax(axis=1),
                ]
            )
            summary = pipe.rate_summary(n, max(1, int(args.stat_sec / (seg * hc.DT))))
            if summary:
                values = [summary[name] for name in rate_names]
                for bar, value in zip(bars, values):
                    bar.set_height(value)
                rate_ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
                oracle_ratio = summary["DQN"] / max(summary["Oracle"], 1e-9) * 100
                random_ratio = summary["Random"] / max(summary["Oracle"], 1e-9) * 100
                rate_title.set_text(
                    f"최근 {args.stat_sec:.0f}s · DQN/오라클 {oracle_ratio:.0f}% · "
                    f"최선 선택 {summary['best_ratio'] * 100:.0f}% · 핸드오버 {summary['handover']:.2f}/구간"
                )
                acc_schedule.set_text(
                    f"RU 선택 정확도 · DQN (최근 {args.stat_sec:.0f} s)\n"
                    f"최선 RU 맞힘 {summary['best_ratio'] * 100:.0f} % · "
                    f"전송률 오라클의 {oracle_ratio:.0f} % (무작위 {random_ratio:.0f} %)"
                )
        else:
            acc_schedule.set_text("RU 선택 정확도\n첫 100 ms 세그먼트를 재생하는 중 …")

        status = "재생" if state["playing"] else "일시정지"
        scale_label = "auto" if scale_source == "auto" else ("pre-scaled X" if scale_source == "pre-scaled X" else Path(scale_source).name)
        representation_label = "legacy X6" if pipe.duplicate_rus else "preserve X3"
        title.set_text(
            f"D-MIMO 데이터셋 데모 — {dataset.name} · {n * hc.DT:6.2f}/{len(pipe.x) * hc.DT:.2f} s · "
            f"{status} · scale {scale:.3g} ({scale_label}, {representation_label}) · Space/←/→/R/Q"
        )
        return []

    def on_key(event):
        key = (event.key or "").lower()
        if key == " ":
            state["playing"] = not state["playing"]
        elif key == "left":
            state["n"] = max(minimum_ticks, int(state["n"]) - int(1.0 / hc.DT))
        elif key == "right":
            state["n"] = min(len(pipe.x), int(state["n"]) + int(1.0 / hc.DT))
        elif key == "home" or key == "r":
            state["n"] = min(len(pipe.x), minimum_ticks)
        elif key == "end":
            state["n"] = len(pipe.x)
        elif key in {"q", "escape"}:
            plt.close(fig)
        update()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    if snapshot:
        update()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(snapshot, dpi=110)
        print(f"[SNAP] {snapshot.resolve()}")
        plt.close(fig)
        return
    animation = FuncAnimation(fig, update, interval=int(1000 / args.fps), cache_frame_data=False)  # noqa: F841
    plt.show()


def print_summary(pipe: OfflinePipeline) -> None:
    n = len(pipe.x)
    nmse = pipe.nmse_db(n, 10**12)
    print("[DONE] NMSE [dB]")
    for name in ("MODEL", "LR", "KF"):
        print(f"       {name:5s}: " + " ".join(f"{value:+6.1f}" for value in nmse[name]))
    rate = pipe.rate_summary(n, 10**9)
    if rate:
        print("[DONE] rate:", {key: round(value, 3) for key, value in rate.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a finite CSI dataset through the channel predictor and RU scheduler, then show the dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", help="run directory, .npz, .npy, or .csv; omit for generated demo CSI")
    parser.add_argument("--key", help="array key inside an NPZ; auto-detected when omitted")
    parser.add_argument("--sample-index", type=int, default=0, help="window index for a three-dimensional X array")
    parser.add_argument("--input-kind", choices=("auto", "raw-csi", "model-x"), default="auto")
    parser.add_argument("--scale", default="auto", help="raw-CSI scale: auto, a number, or scale.json")
    parser.add_argument(
        "--representation",
        choices=("auto", "preserve", "legacy-x6"),
        default="auto",
        help="current three-column X or legacy duplicated six-column X; auto reads the array/scale metadata",
    )
    parser.add_argument("--model-dir", help="directory containing predictor.npz and dqn.npz")
    parser.add_argument("--predictor-model", default="predictor.npz", help="predictor .npz/.pt/.pth model")
    parser.add_argument("--dqn-model", default="dqn.npz", help="RU scheduler .npz/.pt/.pth model")
    parser.add_argument("--context-steps", type=int, default=40, help="GRU context in 25 ms steps")
    parser.add_argument("--min-context-steps", type=int, default=8)
    parser.add_argument("--init-state", type=int, choices=range(6), default=0)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--stat-sec", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot", type=Path, help="save the final dashboard frame to PNG instead of opening a window")
    parser.add_argument("--headless", action="store_true", help="run inference and print metrics without opening the UI")
    parser.add_argument("--save-results", type=Path, help="optional NPZ path for predictions and scheduling outputs")
    args = parser.parse_args()
    if args.context_steps < 1 or args.min_context_steps < 1:
        parser.error("context step counts must be positive")
    if args.min_context_steps > args.context_steps:
        parser.error("--min-context-steps cannot exceed --context-steps")
    if args.fps <= 0 or args.speed <= 0 or args.window_sec <= 0 or args.stat_sec <= 0:
        parser.error("--fps, --speed, --window-sec, and --stat-sec must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        dataset = load_dataset(args.data, args.key, args.sample_index, args.input_kind)
        predictor_fallbacks = (
            ("checkpoints_rnn/best_validation_nmse.pt", "checkpoints_rnn/*.pt")
            if args.predictor_model == "predictor.npz"
            else ()
        )
        dqn_fallbacks = (
            (
                "checkpoints_ru_scheduler/final_best_model.pt",
                "checkpoints_ru_scheduler/best_validation.pt",
                "checkpoints_ru_scheduler/*.pt",
                "checkpoints_double_dqn_ru_windows4s_easiest/*.pt",
                "checkpoints_dqn/*.pt",
            )
            if args.dqn_model == "dqn.npz"
            else ()
        )
        predictor_path = resolve_model(args.predictor_model, args.model_dir, predictor_fallbacks)
        dqn_path = resolve_model(args.dqn_model, args.model_dir, dqn_fallbacks)
        predictor = hc.NumpyPredictor(predictor_path)
        dqn = hc.NumpyDQN(dqn_path)

        scale_spec = args.scale
        checkpoint_scale = dqn.data_meta.get("scale")
        if dataset.input_kind == "raw-csi" and args.scale.lower() == "auto" and checkpoint_scale is not None:
            scale_spec = str(float(checkpoint_scale))
        representation = args.representation
        if representation == "auto" and dqn.timing == "next-segment" and dataset.values.shape[1] == 3:
            representation = "preserve"
        x, scale, scale_source, duplicate_rus = prepare_model_input(dataset, scale_spec, representation)
        if scale_spec != args.scale:
            scale_source = f"{dqn_path.name} data_meta"

        expected_antennas = dqn.data_meta.get("num_antennas")
        actual_scheduler_antennas = 6 if duplicate_rus else x.shape[1]
        if expected_antennas is not None and int(expected_antennas) != actual_scheduler_antennas:
            raise ValueError(
                f"RU scheduler was trained with {expected_antennas} antenna channels, but this input provides "
                f"{actual_scheduler_antennas}. Use matching data/checkpoint or set --representation explicitly."
            )
        ru_antenna_counts = dqn.data_meta.get("ru_antenna_counts")
        expected_counts = [2, 2, 2] if duplicate_rus else [1, 1, 1]
        if ru_antenna_counts is not None and list(ru_antenna_counts) != expected_counts:
            raise ValueError(
                f"RU scheduler uses ru_antenna_counts={ru_antenna_counts}; this dashboard input maps as "
                f"{expected_counts}. Use a matching three-channel dataset/checkpoint."
            )
        action_states = dqn.data_meta.get("action_states")
        expected_states = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
        if action_states is not None and action_states != expected_states:
            raise ValueError(f"RU scheduler action layout {action_states} is incompatible with the dashboard")
        print(
            f"[DATA] {dataset.name} · key={dataset.selected_key} · kind={dataset.input_kind} · "
            f"shape={dataset.values.shape} · {len(x) * hc.DT:.2f} s"
        )
        print(f"[MODEL] predictor={predictor_path}")
        print(f"[MODEL] dqn={dqn_path}")
        print(f"[SCALE] {scale:.6g} ({scale_source})")
        pipeline = OfflinePipeline(x, predictor, dqn, args, duplicate_rus)
        pipeline.run()
        print_summary(pipeline)
        if args.save_results:
            save_results(args.save_results, pipeline, dataset, scale)
        if not args.headless:
            run_gui(pipeline, dataset, scale, scale_source, args, args.snapshot)
        return 0
    except (FileNotFoundError, ImportError, KeyError, IndexError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
