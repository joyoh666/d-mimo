# -*- coding: utf-8 -*-
"""
1017 측정 logs -> 형주형 학습 코드가 읽는 data_ready 형식.

  data_ready/scale.json                     크기 SCALE (학습·실시간 공용)
  data_ready/split.json                      전체 측정 완료 후 정한 episode 단위 train/validation/test split
  data_ready/pred_ws6p00s/X.npz + manifest  채널 예측용  X (N,1200,num_ants)
  data_ready/dqn_ws4p00s/X.npz + manifest   RU 스케줄링용 X (N,800,num_ants) — 전 시나리오, manifest 의 dataset 열로 시험 고정

    python build_dataset.py [--logs ../logs]

SCALE: 학습 시나리오 전체에서 mean|h|² × SCALE² = TARGET_POWER(1e-2).
  형주형 DQN 보상은 noise_power 1e-3 기준 '10 dB' 로 잡혀 있어서 평균 SNR 10 dB 가 되게 맞춘 것.
  실시간에서도 같은 RX_GAIN/TX_GAIN 이면 그대로 쓰고, 바꾸면 live_dashboard.py --auto-scale 로 다시 맞춘다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")   # 윈도우 콘솔(cp949)에서 이모지·한글
except Exception:  # noqa: BLE001
    pass
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hj_common import DATA_READY, DT, default_logs, scale_csi  # noqa: E402

TARGET_POWER = 1e-2


def load_runs(logs: Path):
    runs = {}
    for d in sorted(logs.rglob("2026*_tb_usrp_3tx_S*_ISO3")):
        m = re.search(r"_S(\d\d)_", d.name)
        if not m or not (d / "usrp_sparse_h.npz").exists() or not (d / "scenario_marker.json").exists():
            continue
        sid = f"S{m.group(1)}"
        if sid == "S00":
            continue
        z = np.load(d / "usrp_sparse_h.npz")
        mk = json.loads((d / "scenario_marker.json").read_text(encoding="utf-8"))
        t, c = z["csi_t_usrp"], z["csi"]
        sel = (t >= mk["t0_usrp_est"]) & (t < mk["t1_usrp_est"])
        c = c[sel]
        if np.isnan(c).any():
            raise ValueError(f"{d.name}: NaN in csi")
        runs[sid] = (d.name, c)                                 # 같은 시나리오가 둘이면 뒤(최신) 것
    return runs


def windows(channels, L, stride):
    return [channels[a:a + L] for a in range(0, channels.shape[0] - L + 1, stride)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(default_logs()))
    ap.add_argument("--stride-sec", type=float, default=0.5)
    ap.add_argument("--expected-episodes", type=int, default=75)
    ap.add_argument("--validation-count", type=int, default=5)
    ap.add_argument("--test-count", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=123)
    a = ap.parse_args()

    runs = load_runs(Path(a.logs))
    episode_ids = sorted(runs, key=lambda sid: int(sid[1:]))
    if a.expected_episodes > 0 and len(episode_ids) != a.expected_episodes:
        raise ValueError(
            f"Expected {a.expected_episodes} complete episodes, found {len(episode_ids)}: {episode_ids}"
        )
    if a.validation_count < 1 or a.test_count < 1:
        raise ValueError("validation-count and test-count must both be >= 1")
    if a.validation_count + a.test_count >= len(episode_ids):
        raise ValueError("validation-count + test-count must be smaller than the episode count")

    rng = np.random.default_rng(a.split_seed)
    shuffled = np.asarray(episode_ids, dtype=object)
    rng.shuffle(shuffled)
    test = sorted(shuffled[:a.test_count].tolist(), key=lambda sid: int(sid[1:]))
    validation = sorted(
        shuffled[a.test_count:a.test_count + a.validation_count].tolist(),
        key=lambda sid: int(sid[1:]),
    )
    held_out = set(validation) | set(test)
    train = [sid for sid in episode_ids if sid not in held_out]
    split_by_episode = {
        **{sid: "train" for sid in train},
        **{sid: "validation" for sid in validation},
        **{sid: "test" for sid in test},
    }
    print("runs:", {k: (v[0], v[1].shape) for k, v in runs.items()})
    print("split:", {"train": train, "validation": validation, "test": test})

    p = np.mean([np.mean(np.abs(runs[k][1]) ** 2) for k in train])
    scale = float(np.sqrt(TARGET_POWER / p))
    DATA_READY.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "seed": a.split_seed,
        "expected_episodes": a.expected_episodes,
        "train": train,
        "validation": validation,
        "test": test,
    }
    (DATA_READY / "split.json").write_text(
        json.dumps(split_payload, indent=2), encoding="utf-8"
    )
    (DATA_READY / "scale.json").write_text(json.dumps({
        "scale": scale, "target_power": TARGET_POWER, "train_mean_power_raw": float(p),
        **split_payload,
        "note": "X = scale_csi(csi, scale): 입력 CSI의 안테나 차원을 보존",
    }, indent=2), encoding="utf-8")
    print(f"scale = {scale:.3f}  (raw mean|h|^2 = {p:.3e})")

    stride = int(round(a.stride_sec / DT))

    # 채널 예측: 모든 episode의 6 s 창. 분할은 manifest의 split 열을 학습 코드가 사용한다.
    Xs, rows = [], []
    for k in episode_ids:
        for i, w in enumerate(windows(scale_csi(runs[k][1], scale), 1200, stride)):
            rows.append({
                "sample_id": len(Xs), "dataset": k, "split": split_by_episode[k],
                "run": runs[k][0], "start_tick": i * stride,
            })
            Xs.append(w)
    X = np.stack(Xs).astype(np.complex64)
    out = DATA_READY / "pred_ws6p00s"
    out.mkdir(exist_ok=True)
    np.savez(out / "X.npz", X=X, dt=np.float32(DT))
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print("pred X", X.shape, "->", out / "X.npz")

    # RU 스케줄링: 4 s 창, 전 시나리오 + manifest(dataset = 시나리오 id)
    Xs, rows = [], []
    for k in episode_ids:
        for i, w in enumerate(windows(scale_csi(runs[k][1], scale), 800, stride)):
            rows.append({
                "sample_id": len(Xs), "dataset": k, "split": split_by_episode[k],
                "run": runs[k][0], "start_tick": i * stride,
            })
            Xs.append(w)
    X = np.stack(Xs).astype(np.complex64)
    out = DATA_READY / "dqn_ws4p00s"
    out.mkdir(exist_ok=True)
    np.savez(out / "X.npz", X=X, dt=np.float32(DT), window_sec=np.float32(4.0),
             gain_scales=np.ones(X.shape[-1], np.float32), gain_match_enable=np.int64(0))
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print("dqn X", X.shape, "->", out / "X.npz")


if __name__ == "__main__":
    main()
