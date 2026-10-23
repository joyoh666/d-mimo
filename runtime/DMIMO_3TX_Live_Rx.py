#%%
# #!/usr/bin/env python3
# ============================================================
# D-MIMO 3-TX (ISO) + 1-RX TurtleBot  —  2026-10-17
# ------------------------------------------------------------
# 선배(해웅) FAS_2TX_Turtlebot_Rx_t0.py (d_mimo_1003_hy, 10-05 본시험 판) 를 3 TX 로 늘린 것.
# 실험 순서(pre-roll -> t0 -> 40 s 시나리오 -> 크롭·저장)와 저장 형식은 그대로 두고, 바뀐 것은 두 가지다.
#
#   1) TX 가 셋:  TX0 = 60.2 RF A  (sc18 + PSS/SSS)
#                 TX1 = 60.2 RF B  (sc38)
#                 TX2 = 70.3 RF B  (sc58)      <- 새로 추가 (70.3 두 번째 RF 보드의 TX/RX 포트)
#      RX 는 그대로 70.3 RF A (터틀봇 위 안테나).
#   2) 70.3 이 TX2 와 RX 를 같이 맡으므로 장치를 두 프로세스가 열 수 없다 (X310 은 한 세션만 잡힌다).
#      그래서 60.2 + 70.3 을 MultiUSRP 하나(addr0, addr1)로 열고, 이 스크립트 한 프로세스 안에서
#      TX 3채널 연속 송신 스레드 + RX 1채널 수신 스레드를 같이 돌린다.
#      => 예전처럼 lte_usrp_4sub_tx_copy2.py 를 따로 켜지 않는다. 켜져 있으면 끈다.
#
# 다중 장치를 addrN 으로 묶고 장치별 clock/time 을 맞추는 방식은 후배(승환) channel sounder
# (sh_1017/channel sounder, config/radio_config.py · USRP/usrp_utils.py) 를 따랐다.
# 5 ms 당 CSI 하나(파일럿 3심볼 평균, shape (T, 3))도 후배 형식처럼 npz 의 `csi` 로 같이 남긴다.
#
# 같은 폴더에 있어야 하는 파일: turtle_control.py · turtle_scenarios_40s_random.py · hy_robot_proc.py
#
# [YT 1019] 실시간 데모판 (DMIMO_3TX_Turtlebot_Rx.py 사본). 바뀐 것은 하나뿐이다:
#   5 ms 창마다 세 심볼 평균 CSI (TX 3개) 를 UDP 로 LIVE_UDP (기본 127.0.0.1:5005) 에 쏜다.
#   받는 쪽은 live_demo/live_dashboard.py (형주형 GRU 채널 예측 + DQN RU 스케줄링).
#   패킷: struct '<dii6f' = t_usrp, frame_idx, half(0/1), Re/Im H0, H1, H2.  측정·저장은 원본과 같다.
# ============================================================

import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use(['dark_background', 'fast'])

import uhd
import scipy.signal
import socket, struct                                           # [YT 1019] 실시간 송출

LIVE_UDP = os.environ.get("LIVE_UDP", "127.0.0.1:5005")         # [YT 1019] 빈 문자열이면 송출 안 함
_live_sock = None
if LIVE_UDP:
    _live_host, _live_port = LIVE_UDP.rsplit(":", 1)
    _live_addr = (_live_host, int(_live_port))
    _live_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_LIVE_FMT = struct.Struct("<dii6f")

# NO_ROBOT=1 이면 ROS/터틀봇 없이 같은 순서(pre-roll -> t0 -> DURATION)로 채널만 받는다 (정지 점검용).
NO_ROBOT = os.environ.get("NO_ROBOT", "0") == "1"
# ROBOT_PROC=1(기본): 터틀봇 제어를 자식 프로세스(hy_robot_proc.py)로 뗀다 (10-04 실측: 한 프로세스면 위치 20 -> 5 Hz).
ROBOT_PROC = (os.environ.get("ROBOT_PROC", "1") == "1") and not NO_ROBOT
TOTAL_DURATION = float(os.environ.get("DURATION", "15"))       # 시나리오 파일의 15 s 와 같아야 한다 (NO_ROBOT 이면 아무 값)
SCENARIOS = range(1, 76)


# =========================
# USER PARAMETERS
# =========================
SCENARIO_ID = int(os.environ.get("SCENARIO_ID", "1"))
if SCENARIO_ID not in SCENARIOS:
    raise ValueError("SCENARIO_ID must be 1..75")
ANT_TAG = os.environ.get("ANT_TAG", "ISO3").strip().replace(" ", "_")

PRE_ROLL_SECONDS  = 2.0
POST_ROLL_SECONDS = 1.0
RUN_SECONDS   = PRE_ROLL_SECONDS + TOTAL_DURATION + POST_ROLL_SECONDS + 1.0
ROBOT_LOG_HZ  = float(os.environ.get("ROBOT_LOG_HZ", "5"))
SAVE_ROOT     = Path(os.environ.get("SAVE_ROOT", f"./logs/{datetime.now():%Y%m%d}"))   # [YT 1019] 날짜 폴더 logs/YYYYMMDD/
NOTE          = os.environ.get("NOTE", "")                                          # [YT 1019] index.csv 메모 칸
EXP_TAG       = f"tb_usrp_3tx_S{SCENARIO_ID:02d}_{ANT_TAG}"

WINDOW_SEC = 0.005                 # 5 ms = 반 프레임 = 70 OFDM 심볼
SYMS_PER_WINDOW = [1, 2, 3]        # 5 ms 마다 이 세 심볼에서 H 를 잰다 (프레임 첫 슬롯의 sync 심볼 5/6 은 피함)

WAIT_POSE_TIMEOUT_SEC = 30.0
RING_SECONDS = float(os.environ.get("RING_SECONDS", "4.0"))
RECV_CHUNK = int(os.environ.get("RECV_CHUNK", "20000"))
RX_TIMEOUT_SEC = 0.20
POSE_QUERY_LAG_SEC = 0.10


# =========================
# 장치 · RF
# =========================
carrier_frequency = float(os.environ.get("FC", "2.5e9"))

TX_ADDR  = os.environ.get("TX_ADDR",  "192.168.60.2")         # mboard 0: TX0 (RF A) · TX1 (RF B)
TRX_ADDR = os.environ.get("TRX_ADDR", os.environ.get("RX_ADDR", "192.168.70.3"))   # mboard 1: TX2 (RF B) · RX (RF A)
TX2_SUBDEV = os.environ.get("TX2_SUBDEV", "B:0")               # 70.3 두 번째 RF 보드
RX_SUBDEV  = os.environ.get("RX_SUBDEV", "A:0")                # 70.3 첫 번째 RF 보드 (터틀봇 안테나)
TX_ANT = os.environ.get("TX_ANT", "TX/RX")                     # 각 보드의 첫 번째 구멍
RX_ANT = os.environ.get("RX_ANT", "TX/RX")

# 기준 클럭: 지금 배선은 60.2 REF/PPS OUT -> 70.3 REF/PPS IN (10-16 cross-pair 와 같음).
# 옥토클럭에 둘 다 물리면 CLOCK_SRC0=external TIME_SRC0=external.
CLOCK_SRC0 = os.environ.get("CLOCK_SRC0", "internal")
TIME_SRC0  = os.environ.get("TIME_SRC0",  "internal")
CLOCK_SRC1 = os.environ.get("CLOCK_SRC1", os.environ.get("CLOCK_SRC", "external"))
TIME_SRC1  = os.environ.get("TIME_SRC1",  os.environ.get("TIME_SRC",  "external"))

Tx_gain  = float(os.environ.get("TX_GAIN", "20"))
Tx_gains = [float(os.environ.get(f"TX_GAIN{i}", Tx_gain)) for i in range(3)]
Rx_gain  = float(os.environ.get("RX_GAIN", "20"))
POWER = 1/8                                                    # 선배 TX 와 같음 (진폭 0.35)

bandwidth_options = {
    1.4:    (72,    128),
    3:      (180,   256),
    5:      (300,   512),
    10:     (600,   1024),
    15:     (900,   1536),
    20:     (1200,  2048),
    50:     (3000,  4096)
}
BANDWIDTH = 1.4
N, FFT_SIZE = bandwidth_options[BANDWIDTH]

delta_f = 15e3
normal_CP_time = 4.7e-6
first_CP_time  = 5.2e-6

N_PSS = 62
num_symbols_per_slot = 7
num_slot_per_subframe = 2
num_subframe_per_frame = 10

sampling_rate = delta_f * FFT_SIZE

normal_CP_length = round(normal_CP_time * sampling_rate)
first_CP_length  = round(first_CP_time  * sampling_rate)

slot_length = (
    FFT_SIZE * num_symbols_per_slot +
    normal_CP_length * (num_symbols_per_slot - 1) +
    first_CP_length
)
frame_length = slot_length * num_slot_per_subframe * num_subframe_per_frame  # 10 ms

SYMS_PER_FRAME = num_subframe_per_frame * num_slot_per_subframe * num_symbols_per_slot  # 140
frame_duration_sec = frame_length / sampling_rate
half_frame_samples = frame_length // 2


# =========================
# 파일럿 (TX 별 전용 부반송파, 매 심볼)
# =========================
sc_list = [int(s) for s in os.environ.get("SC_LIST", "18,38,58").split(",")]   # TX0, TX1, TX2
NUM_TX = len(sc_list)
if NUM_TX != 3:
    raise ValueError("SC_LIST 는 부반송파 세 개 (TX0,TX1,TX2)")
if any(not (0 <= s < N) for s in sc_list) or len(set(sc_list)) != NUM_TX:
    raise ValueError(f"SC_LIST 는 0..{N-1} 안의 서로 다른 값")
sc_indices = np.array(sc_list, dtype=np.int32)

PILOT_SEED = 2026
const = (np.array([1+1j, 1-1j, -1+1j, -1-1j], np.complex64) / np.sqrt(2))
pilot_block = np.random.default_rng(PILOT_SEED).choice(
    const,
    size=(num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot - 1, NUM_TX),   # (10,2,6,3)
    replace=True
).astype(np.complex64)


def get_zc_sequence(Nseq, q=25):
    m = np.arange(Nseq)
    seq = -np.pi * q * m * (m + 1) / Nseq
    return np.cos(seq) + 1j*np.sin(seq)


def sss_sequence(Nseq, seed):
    bits = np.random.default_rng(seed).integers(0, 2, size=Nseq)
    return (2*bits - 1).astype(np.complex64)


zc_seq = get_zc_sequence(N_PSS, 25)
pss_start_idx = N - (N_PSS+1) - ((N - (N_PSS+1))//2)
pss = np.zeros(N, np.complex64)
pss[pss_start_idx:pss_start_idx+N_PSS] = zc_seq.astype(np.complex64)
sss_first = np.zeros(N, np.complex64)
sss_first[pss_start_idx:pss_start_idx+N_PSS] = sss_sequence(N_PSS, 0)
dmrs = get_zc_sequence(N, 1).astype(np.complex64)


def build_resource_maps():
    """(NUM_TX, SF, SL, SYM, N).  TX k 는 sc_list[k] 에만 실림. TX0 만 SF0/SL0/SYM5..6 에 SSS/PSS, 나머지는 그때 침묵."""
    rm = np.zeros((NUM_TX, num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, N), np.complex64)
    for k, sc in enumerate(sc_list):
        rm[k, :, :, 0, sc] = dmrs[sc]
        rm[k, :, :, 1:, sc] = pilot_block[..., k]
    rm[0, 0, 0, 5, :] = sss_first
    rm[0, 0, 0, 6, :] = pss
    rm[1:, 0, 0, 5, :] = 0
    rm[1:, 0, 0, 6, :] = 0
    return rm


def build_waveform_from_resource_maps(rm):
    fft_symbols = np.zeros(rm.shape[:-1] + (FFT_SIZE,), np.complex64)
    fft_symbols[..., 1:N//2+1] = rm[..., N//2:]
    fft_symbols[..., -(N//2):] = rm[..., :N//2]
    td = np.fft.ifft(fft_symbols, axis=-1, norm='ortho').astype(np.complex64)
    first = np.concatenate([td[..., 0:1, -first_CP_length:], td[..., 0:1, :]], axis=-1)
    other = np.concatenate([td[..., 1:, -normal_CP_length:], td[..., 1:, :]], axis=-1)
    lead = rm.shape[:-2]                                       # (NUM_TX, SF, SL)
    w = np.concatenate([first.reshape(lead + (-1,)), other.reshape(lead + (-1,))], axis=-1)
    w *= np.sqrt(POWER)
    return w.reshape(rm.shape[0], -1).astype(np.complex64)


resource_maps = build_resource_maps()
waveform = build_waveform_from_resource_maps(resource_maps)    # (3, frame_length)
assert waveform.shape == (NUM_TX, frame_length)


def X_known_for_symbol(sf, sl, sym):
    return resource_maps[:, sf, sl, sym, sc_indices].diagonal().astype(np.complex64)


def ofdm_td_with_cp_from_N(v):
    X = np.zeros(FFT_SIZE, np.complex64)
    X[1:N//2+1] = v[N//2:]
    X[-(N//2):] = v[:N//2]
    x = np.fft.ifft(X, norm='ortho').astype(np.complex64)
    return np.concatenate([x[-normal_CP_length:], x], axis=0)


sync_ref_td = np.concatenate([ofdm_td_with_cp_from_N(sss_first), ofdm_td_with_cp_from_N(pss)], axis=0)
ss_start_idx = FFT_SIZE * 5 + first_CP_length + normal_CP_length * 4     # 프레임 안 SYM5 CP 시작

cp_len_per_sym = np.empty(SYMS_PER_FRAME, dtype=np.int32)
sym_start_offsets = np.empty(SYMS_PER_FRAME, dtype=np.int64)
off = 0
idx = 0
for sf in range(num_subframe_per_frame):
    for sl in range(num_slot_per_subframe):
        for sym in range(num_symbols_per_slot):
            sym_start_offsets[idx] = off
            cp_len_per_sym[idx] = first_CP_length if sym == 0 else normal_CP_length
            idx += 1
            off += (first_CP_length if sym == 0 else normal_CP_length) + FFT_SIZE
assert idx == SYMS_PER_FRAME and off == frame_length


# =========================
# USRP: 60.2 + 70.3 을 MultiUSRP 하나로
# =========================
def sensor_bool(fn, *a):
    try:
        return fn(*a).to_bool()
    except Exception:
        return None


usrp = uhd.usrp.MultiUSRP(f"addr0={TX_ADDR},addr1={TRX_ADDR}")
NUM_MB = usrp.get_num_mboards()
if NUM_MB != 2:
    raise RuntimeError(f"장치가 {NUM_MB} 대만 잡혔다 (60.2·70.3 둘 다 ping 되는지, 다른 터미널에서 USRP 를 잡고 있지 않은지)")

for mb, (cs, ts) in enumerate([(CLOCK_SRC0, TIME_SRC0), (CLOCK_SRC1, TIME_SRC1)]):
    usrp.set_clock_source(cs, mb)
    usrp.set_time_source(ts, mb)
time.sleep(1.0)
ref_locked = [sensor_bool(usrp.get_mboard_sensor, "ref_locked", mb) for mb in range(NUM_MB)]
for mb, addr in enumerate([TX_ADDR, TRX_ADDR]):
    print(f"[MB{mb}] {addr}  clock={usrp.get_clock_source(mb)}  time={usrp.get_time_source(mb)}  ref_locked={ref_locked[mb]}")
if ref_locked[1] is False:
    print("\n!!!! 70.3 이 기준 클럭에 안 잠겼다 — 60.2 REF OUT -> 70.3 REF IN (10 MHz) 선을 확인.")
    print("!!!! 이대로면 TX0/TX1 과 TX2 의 상대위상이 돌고, RX 프레임 동기도 수 초 안에 벗어난다.\n")

# 두 장치 시간을 같은 PPS 에지에 0 으로 (60.2 PPS OUT -> 70.3 PPS IN).
usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
time.sleep(0.2)
t_mb = [usrp.get_time_now(mb).get_real_secs() for mb in range(NUM_MB)]
mb_time_diff = abs(t_mb[1] - t_mb[0])
print(f"[TIME] MB0 {t_mb[0]:.6f} s · MB1 {t_mb[1]:.6f} s  (차 {mb_time_diff*1e3:.3f} ms — 읽는 사이 지연 포함, 1 ms 넘으면 PPS 선 확인)")
if mb_time_diff > 1e-3:
    print("!!!! 두 장치 시간이 안 맞는다 — TX2 프레임이 TX0 프레임과 어긋나서 TX2 H 가 깨진다. PPS 선 확인.")

# 채널 번호: TX  ch0 = 60.2 A · ch1 = 60.2 B · ch2 = 70.3 B
#            RX  ch0 = 60.2 A (안 씀) · ch1 = 70.3 A
usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"), 0)
usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(TX2_SUBDEV), 1)
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:0"), 0)
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(RX_SUBDEV), 1)
TX_CHANNELS = [0, 1, 2]
RX_CH = 1
if usrp.get_tx_num_channels() != 3 or usrp.get_rx_num_channels() != 2:
    raise RuntimeError(f"채널 수가 이상하다: TX {usrp.get_tx_num_channels()} (3 이어야) · RX {usrp.get_rx_num_channels()} (2 이어야)")

lo_offset = sampling_rate / 2 + 1e6                            # 선배 TX 와 같음: TX LO 누설을 대역 밖으로
for i, c in enumerate(TX_CHANNELS):
    usrp.set_tx_antenna(TX_ANT, c)
    usrp.set_tx_rate(sampling_rate, c)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(carrier_frequency, lo_offset), c)
    usrp.set_tx_gain(Tx_gains[i], c)
usrp.set_rx_antenna(RX_ANT, RX_CH)
usrp.set_rx_rate(sampling_rate, RX_CH)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(carrier_frequency), RX_CH)
usrp.set_rx_gain(Rx_gain, RX_CH)
time.sleep(0.5)

TX_NAMES = [f"TX0 {TX_ADDR} A", f"TX1 {TX_ADDR} B", f"TX2 {TRX_ADDR} {TX2_SUBDEV[0]}"]
for i, c in enumerate(TX_CHANNELS):
    print(f"[{TX_NAMES[i]}] sc{sc_list[i]}  {usrp.get_tx_freq(c)/1e9:.6f} GHz  gain {usrp.get_tx_gain(c):.1f}  "
          f"rate {usrp.get_tx_rate(c):.1f}  lo_locked {sensor_bool(usrp.get_tx_sensor, 'lo_locked', c)}")
fs_actual = float(usrp.get_rx_rate(RX_CH))                     # X310: 1.92 MHz 요청 -> 1.923077 MHz
print(f"[RX  {TRX_ADDR} {RX_SUBDEV}] {usrp.get_rx_freq(RX_CH)/1e9:.6f} GHz  gain {usrp.get_rx_gain(RX_CH):.1f}  "
      f"rate {fs_actual:.1f} (nominal {sampling_rate:.1f})  lo_locked {sensor_bool(usrp.get_rx_sensor, 'lo_locked', RX_CH)}")

tx_args = uhd.usrp.StreamArgs("fc32", "sc16")
tx_args.channels = TX_CHANNELS
tx_streamer = usrp.get_tx_stream(tx_args)
rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
rx_args.channels = [RX_CH]
rx_streamer = usrp.get_rx_stream(rx_args)


# =========================
# TX: 10 ms 프레임을 끊김 없이 반복 (세 채널을 한 스트리머로 -> 같은 시각 정렬)
# =========================
TX_BLOCK_FRAMES = int(os.environ.get("TX_BLOCK_FRAMES", "10"))   # send 한 번 = 100 ms
tx_block = np.ascontiguousarray(np.tile(waveform, (1, TX_BLOCK_FRAMES)))
tx_stop = threading.Event()
tx_started = threading.Event()
tx_state = {"underflow": 0, "seq_err": 0, "time_err": 0, "error": None, "t_start": None}


def tx_thread_loop():
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = False
    md.has_time_spec = True
    t_start = usrp.get_time_now(0).get_real_secs() + 1.0
    md.time_spec = uhd.types.TimeSpec(t_start)
    tx_state["t_start"] = t_start
    L = tx_block.shape[1]
    try:
        while not tx_stop.is_set():
            sent = 0
            while sent < L and not tx_stop.is_set():
                n = tx_streamer.send(np.ascontiguousarray(tx_block[:, sent:]) if sent else tx_block, md)
                if n > 0:
                    sent += n
                    md.start_of_burst = False
                    md.has_time_spec = False
            tx_started.set()
    except Exception as e:                                     # noqa: BLE001
        tx_state["error"] = repr(e)
        print("[TX] 송신 스레드 오류:", repr(e))
    finally:
        try:
            md2 = uhd.types.TXMetadata()
            md2.end_of_burst = True
            tx_streamer.send(np.zeros((NUM_TX, 0), dtype=np.complex64), md2)
        except Exception:
            pass


def tx_async_loop():
    """언더플로(U)·순서 오류를 센다. 0 이 아니면 그 구간 파형이 끊겼다."""
    try:
        am = uhd.types.TXAsyncMetadata()
        E = uhd.types.TXMetadataEventCode
    except Exception:                                          # noqa: BLE001  (이 UHD 판에 없으면 세지 않는다)
        tx_state["underflow"] = None
        return
    while not tx_stop.is_set():
        try:
            if not tx_streamer.recv_async_msg(am, 0.1):
                continue
        except Exception:
            return
        ev = am.event_code
        if ev in (E.underflow, E.underflow_in_packet):
            tx_state["underflow"] += 1
        elif ev in (E.seq_error, E.seq_error_in_packet):
            tx_state["seq_err"] += 1
        elif ev == E.time_error:
            tx_state["time_err"] += 1


tx_thread = threading.Thread(target=tx_thread_loop, daemon=True)
tx_async_thread = threading.Thread(target=tx_async_loop, daemon=True)


# =========================
# ROS2 + Turtlebot (선배 코드와 같음: 로봇은 자식 프로세스)
# =========================
if not NO_ROBOT and not ROBOT_PROC:
    raise RuntimeError("이 스크립트는 ROBOT_PROC=1(기본) 또는 NO_ROBOT=1 만 지원한다")

robot_proc = None
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = SAVE_ROOT / f"{stamp}_{EXP_TAG}"
out_dir.mkdir(parents=True, exist_ok=True)

if ROBOT_PROC:
    import subprocess, sys as _sys
    _here = Path(__file__).resolve().parent
    print("SLAM 데이터 수신 대기...")
    robot_proc = subprocess.Popen([_sys.executable, "-u", str(_here / "hy_robot_proc.py"), str(SCENARIO_ID),
                                   str((out_dir / "scenario").resolve()), str(PRE_ROLL_SECONDS)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=str(_here),
                                  encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    for _line in robot_proc.stdout:
        _line = _line.rstrip("\n")
        if _line == "@@READY":
            break
        if _line.startswith("@@ERROR"):
            raise RuntimeError(_line)
        print(_line)
    else:
        raise RuntimeError("로봇 프로세스가 READY 전에 끝났다")
    print("Pose 수신 시작 OK.")
else:
    print("[NO_ROBOT] 터틀봇 없이 진행")


# =========================
# RX: 연속 수신 -> 링 버퍼 (선배 코드와 같음)
# =========================
ring_len = int(RING_SECONDS * sampling_rate)
ring = np.zeros(ring_len, dtype=np.complex64)
ring_peak = 0.0
gap_log = []
rx_lock = threading.Lock()
rx_stop = threading.Event()
total_samples = 0
ring_write_pos = 0
time_usrp0 = None
time_os0 = None
overflow_count = 0
other_err_count = 0


def ring_write(x):
    global ring_write_pos
    n = x.size
    end = ring_write_pos + n
    if end <= ring_len:
        ring[ring_write_pos:end] = x
    else:
        k = ring_len - ring_write_pos
        ring[ring_write_pos:] = x[:k]
        ring[:end - ring_len] = x[k:]
    ring_write_pos = (ring_write_pos + n) % ring_len


def get_samples_abs(abs_start, n):
    with rx_lock:
        abs_end = abs_start + n
        min_abs = total_samples - ring_len
        if abs_start < min_abs or abs_end > total_samples:
            raise RuntimeError("Requested samples not in ring buffer")
        start_idx = (ring_write_pos + abs_start - min_abs) % ring_len
        if start_idx + n <= ring_len:
            return ring[start_idx:start_idx+n].copy()
        k = ring_len - start_idx
        return np.concatenate([ring[start_idx:].copy(), ring[:n-k].copy()], axis=0)


def rx_thread_loop():
    global total_samples, time_usrp0, time_os0, overflow_count, other_err_count, ring_peak, ring_write_pos
    metadata = uhd.types.RXMetadata()
    buf = np.zeros((1, max(rx_streamer.get_max_num_samps(), RECV_CHUNK)), dtype=np.complex64)
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)
    first = True
    try:
        while not rx_stop.is_set():
            got = rx_streamer.recv(buf, metadata, timeout=RX_TIMEOUT_SEC)
            if got > 0:
                x = buf[0, :got].copy()
                if first:
                    time_usrp0 = metadata.time_spec.get_real_secs()
                    time_os0 = time.time()
                    first = False
                else:
                    # overflow 로 빠진 표본은 USRP 타임스탬프로 위치를 다시 구해 0 으로 채운다 (격자 유지)
                    expected = int(round((metadata.time_spec.get_real_secs() - time_usrp0) * fs_actual))
                    gap = expected - total_samples
                    if gap > 0:
                        with rx_lock:
                            if gap >= ring_len:
                                ring[:] = 0
                                ring_write_pos = (ring_write_pos + gap) % ring_len
                            else:
                                ring_write(np.zeros(gap, dtype=np.complex64))
                            total_samples += gap
                        gap_log.append((float(total_samples / fs_actual), int(gap)))
                        print(f"[RX] 표본 {gap} 개 ({gap / fs_actual * 1e3:.1f} ms) 빠짐 — t={total_samples / fs_actual:.2f} s, 0 으로 채움")
                with rx_lock:
                    ring_write(x)
                    total_samples += got
                ring_peak = max(ring_peak, float(np.abs(x[::16]).max()))
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    overflow_count += 1
                    try:
                        c2 = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
                        c2.stream_now = True
                        rx_streamer.issue_stream_cmd(c2)
                    except Exception as e:                     # noqa: BLE001
                        print("[RX] 재시작 명령 실패:", repr(e))
                elif metadata.error_code != uhd.types.RXMetadataErrorCode.timeout:
                    other_err_count += 1
    finally:
        rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))


rx_thread = threading.Thread(target=rx_thread_loop, daemon=True)


# =========================
# 호스트 시계 <-> USRP 시계 (t0 환산용, RX 장치 = MB1 기준)
# =========================
def calibrate_host_usrp_offset(n=31):
    obs = []
    for _ in range(n):
        h0 = time.monotonic()
        u = usrp.get_time_now(1).get_real_secs()
        h1 = time.monotonic()
        obs.append((h1 - h0, u - 0.5 * (h0 + h1)))
        time.sleep(0.002)
    obs.sort(key=lambda z: z[0])
    keep = obs[:max(5, n // 5)]
    return {
        "offset_usrp_minus_monotonic": float(np.median([x[1] for x in keep])),
        "best_rtt_s": float(obs[0][0]),
        "median_kept_rtt_s": float(np.median([x[0] for x in keep])),
        "num_samples": int(n),
        "num_kept": len(keep),
    }


def monotonic_to_usrp(t_mono, mapping):
    return float(t_mono + mapping["offset_usrp_minus_monotonic"])


# =========================
# H 추정
# =========================
def estimate_H_for_symbol(frame_start_abs, sym_in_frame):
    cp = int(cp_len_per_sym[sym_in_frame])
    fft_start_abs = frame_start_abs + int(sym_start_offsets[sym_in_frame]) + cp
    td = get_samples_abs(fft_start_abs, FFT_SIZE)
    t_usrp = time_usrp0 + fft_start_abs / fs_actual
    t_os = time_os0 + fft_start_abs / fs_actual
    H = np.full(NUM_TX, np.nan + 1j*np.nan, dtype=np.complex64)
    if not np.any(td):                                         # overflow 로 0 을 채운 구간
        return H, t_usrp, t_os
    Xf = np.fft.fft(td, norm='ortho')
    rmN = np.empty(N, dtype=np.complex64)
    rmN[N//2:] = Xf[1:N//2+1]
    rmN[:N//2] = Xf[-(N//2):]
    Y = rmN[sc_indices]
    rem = sym_in_frame % (num_slot_per_subframe * num_symbols_per_slot)
    Xk = X_known_for_symbol(sym_in_frame // (num_slot_per_subframe * num_symbols_per_slot),
                            rem // num_symbols_per_slot, rem % num_symbols_per_slot)
    m = np.abs(Xk) > 1e-12
    H[m] = Y[m] / Xk[m]
    return H, t_usrp, t_os


def initial_frame_sync():
    corr_len = int(2 * frame_length + len(sync_ref_td) + 2000)
    while True:
        with rx_lock:
            have = total_samples
        if have >= corr_len and time_usrp0 is not None:
            break
        time.sleep(0.01)
    with rx_lock:
        end_abs = total_samples
    start_abs = max(0, end_abs - corr_len)
    x = get_samples_abs(start_abs, end_abs - start_abs)
    x = x - np.mean(x)
    corr = np.abs(scipy.signal.correlate(x, sync_ref_td, mode='valid', method='fft'))
    pk = int(np.argmax(corr))
    sync_quality = float(corr[pk] / (np.median(corr) + 1e-12))
    frame_start_abs = start_abs + pk - ss_start_idx
    if frame_start_abs < 0:
        frame_start_abs += int(np.ceil(-frame_start_abs / frame_length)) * frame_length
    return int(frame_start_abs), sync_quality


# =========================
# MAIN
# =========================
print("\n====================")
print("EXPERIMENT START  (3 TX ISO -> 1 RX turtlebot)")
print("RUN_SECONDS     :", RUN_SECONDS)
print("SCENARIO_ID     :", SCENARIO_ID, "(NO_ROBOT)" if NO_ROBOT else "")
print("Subcarriers     :", dict(zip(["TX0", "TX1", "TX2"], sc_list)))
print("TX gains        :", Tx_gains, " RX gain:", Rx_gain)
print("Save dir        :", out_dir)
print("====================\n")

tx_async_thread.start()
tx_thread.start()
if not tx_started.wait(timeout=5.0):
    raise RuntimeError(f"TX 가 시작되지 않았다: {tx_state['error']}")
while usrp.get_time_now(0).get_real_secs() < tx_state["t_start"] + 0.2:   # 송신이 실제로 나가기 시작한 뒤 수신
    time.sleep(0.02)
print(f"[TX] 3채널 송신 중 (t_start {tx_state['t_start']:.3f} s)")
rx_thread.start()

print("[SYNC] Waiting for initial frame sync...")
frame0_start_abs, sync_quality = initial_frame_sync()
print(f"[SYNC] frame0_start_abs = {frame0_start_abs} · 상관 피크/중앙값 = {sync_quality:.1f} (10 아래면 TX0 신호가 약하거나 없다)")

clock_map = calibrate_host_usrp_offset()
print("[SYNC] monotonic->USRP offset =", clock_map["offset_usrp_minus_monotonic"], " best RTT", clock_map["best_rtt_s"], "s")

with rx_lock:
    now_abs = total_samples
k0 = int(np.floor((now_abs - frame0_start_abs) / half_frame_samples))
next_half_start_abs = frame0_start_abs + max(0, k0) * half_frame_samples

usrp_rows = []   # (t_os, t_usrp, frame_idx, sym_in_frame, H_sc18, H_sc38, H_sc58)
scenario_result = {}


def _scenario_start_callback(t0_wall, t0_mono):
    return {"t0_usrp_est": monotonic_to_usrp(t0_mono, clock_map), "clock_map": clock_map}


def run_scenario_norobot(countdown, start_callback):
    time.sleep(countdown)
    t0_wall = time.time(); t0_mono = time.monotonic()
    cb = start_callback(t0_wall, t0_mono)
    print(f"[T0] (NO_ROBOT) started at host wall time {t0_wall:.6f}")
    time.sleep(TOTAL_DURATION)
    return {"scenario_id": SCENARIO_ID, "t0_wall": t0_wall, "t0_monotonic": t0_mono,
            "duration_s": TOTAL_DURATION, "callback_data": cb, "aborted": False, "no_robot": True}


def run_scenario_proc(countdown, start_callback):
    robot_proc.stdin.write("GO\n"); robot_proc.stdin.flush()
    cbd, ev = None, None
    for line in robot_proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("@@T0 "):
            _, w, m = line.split()
            cbd = start_callback(float(w), float(m))
        elif line.startswith("@@DONE "):
            ev = json.loads(line[7:]); break
        elif line.startswith("@@ERROR"):
            raise RuntimeError(line)
        else:
            print(line)
    if ev is None:
        raise RuntimeError("로봇 프로세스가 DONE 없이 끝났다")
    ev["callback_data"] = cbd
    return ev


def _scenario_worker():
    try:
        ev = run_scenario_norobot(PRE_ROLL_SECONDS, _scenario_start_callback) if NO_ROBOT \
            else run_scenario_proc(PRE_ROLL_SECONDS, _scenario_start_callback)
        scenario_result.update(ev)
    except Exception as e:                                     # noqa: BLE001
        scenario_result["error"] = repr(e)
        print("[SCENARIO] ERROR:", repr(e))


scenario_thread = threading.Thread(target=_scenario_worker, daemon=True)
scenario_thread.start()
t_run_start = time.monotonic()

try:
    while time.monotonic() - t_run_start < RUN_SECONDS:
        half_idx_global = int((next_half_start_abs - frame0_start_abs) // half_frame_samples)
        frame_idx = half_idx_global // 2
        window_start_sym = 0 if (half_idx_global % 2 == 0) else 70
        target_syms = [window_start_sym + s for s in SYMS_PER_WINDOW]
        frame_start_abs = next_half_start_abs - int(sym_start_offsets[window_start_sym])
        last = max(target_syms)
        required_abs_end = frame_start_abs + int(sym_start_offsets[last]) + int(cp_len_per_sym[last]) + FFT_SIZE

        skip = False
        while True:
            with rx_lock:
                have = total_samples
            if required_abs_end <= have and frame_start_abs >= have - ring_len:
                break
            if frame_start_abs < have - ring_len:              # 처리가 밀려 링 밖으로 나간 창은 건너뛴다
                skip = True
                break
            time.sleep(0.005)
        try:
            if skip:
                raise RuntimeError
            est = [estimate_H_for_symbol(frame_start_abs, s_) for s_ in target_syms]
        except RuntimeError:
            with rx_lock:
                now_abs2 = total_samples
            next_half_start_abs = frame0_start_abs + max(0, int(np.floor((now_abs2 - frame0_start_abs) / half_frame_samples))) * half_frame_samples
            continue
        for s_, (H, t_usrp, t_os) in zip(target_syms, est):
            usrp_rows.append((float(t_os), float(t_usrp), int(frame_idx), int(s_), *[complex(v) for v in H]))
        if _live_sock is not None:                              # [YT 1019] 5 ms CSI = 세 심볼 평균 (저장되는 csi 와 같은 정의)
            _h = np.nanmean(np.array([e_[0] for e_ in est], dtype=np.complex64), axis=0)
            _t = float(np.mean([e_[1] for e_ in est]))
            try:
                _live_sock.sendto(_LIVE_FMT.pack(_t, int(frame_idx), int(window_start_sym >= 70),
                                                 *[float(v) for c_ in _h for v in (c_.real, c_.imag)]), _live_addr)
            except OSError:
                pass
        next_half_start_abs += half_frame_samples
except KeyboardInterrupt:
    print("\n[INFO] KeyboardInterrupt: stopping early.")
finally:
    rx_stop.set()
    time.sleep(0.2)
    tx_stop.set()

scenario_thread.join(timeout=2.0)
if robot_proc is not None:
    try:
        robot_proc.wait(timeout=5.0)
    except Exception:
        robot_proc.kill()
tx_thread.join(timeout=2.0)

robot_rows = []
_raw = out_dir / "scenario" / "robot_states_raw.csv"
if _raw.exists():
    robot_rows = [tuple(r) for r in pd.read_csv(_raw).itertuples(index=False, name=None)]

print(f"\n[RX] overflow_count = {overflow_count}, other_err_count = {other_err_count}")
print(f"[RX] 빠진 구간 {len(gap_log)} 개 · 합계 {sum(g for _, g in gap_log) / fs_actual * 1e3:.1f} ms (그 구간 H 는 NaN)")
print(f"[RX] peak |x| = {ring_peak:.3f}  (1.0 근처면 포화 -> RX_GAIN 내림 · 0.01 아래면 올림)")
print(f"[TX] underflow = {tx_state['underflow']} · seq_err = {tx_state['seq_err']} · time_err = {tx_state['time_err']}"
      f"{'' if tx_state['error'] is None else ' · 오류 ' + tx_state['error']}")
print(f"[RX] collected rows = {len(usrp_rows)} (each row -> H_sc{sc_list[0]}, H_sc{sc_list[1]}, H_sc{sc_list[2]})")
if not usrp_rows:
    raise RuntimeError("No USRP measurements collected. Check TX, timing sync, and RF.")


# =========================
# SAVE (선배 형식 + H 열 하나 더 + 5 ms CSI)
# =========================
robot_df = pd.DataFrame(robot_rows, columns=["t_os", "x", "y", "z", "vx", "vy", "vz"])
robot_csv = out_dir / "turtlebot_states.csv"
robot_df.to_csv(robot_csv, index=False)

h_cols = [f"H_sc{sc}" for sc in sc_list]
usrp_df = pd.DataFrame(usrp_rows, columns=["t_os", "t_usrp", "frame_idx", "sym_in_frame"] + h_cols)
usrp_csv = out_dir / "usrp_sparse_h.csv"
usrp_df.to_csv(usrp_csv, index=False)

scenario_usrp_csv = None
t0_usrp_est = t0_wall = None
if scenario_result.get("callback_data"):
    t0_usrp_est = float(scenario_result["callback_data"]["t0_usrp_est"])
    t0_wall = float(scenario_result["t0_wall"])
    m_h = (usrp_df["t_usrp"] >= t0_usrp_est) & (usrp_df["t_usrp"] < t0_usrp_est + TOTAL_DURATION)
    d = usrp_df.loc[m_h].copy(); d.insert(0, "t_rel", d["t_usrp"] - t0_usrp_est)
    scenario_usrp_csv = out_dir / "usrp_scenario_window_40s.csv"
    d.to_csv(scenario_usrp_csv, index=False)
    m_r = (robot_df["t_os"] >= t0_wall) & (robot_df["t_os"] < t0_wall + TOTAL_DURATION)
    r = robot_df.loc[m_r].copy(); r.insert(0, "t_rel", r["t_os"] - t0_wall)
    r.to_csv(out_dir / "turtle_scenario_window_40s.csv", index=False)
    with open(out_dir / "scenario_marker.json", "w") as f:
        json.dump({
            "scenario_id": SCENARIO_ID, "scenario_name": f"bounded_random_S{SCENARIO_ID:02d}", "ant_tag": ANT_TAG,
            "t0_wall": t0_wall, "t0_usrp_est": t0_usrp_est, "t1_usrp_est": t0_usrp_est + TOTAL_DURATION,
            "duration_s": TOTAL_DURATION, "clock_map": clock_map, "scenario_event": scenario_result,
        }, f, indent=2, default=str)
else:
    print("[WARN] No scenario t0 marker was recorded; 40-s cropped files were not generated.")

H = np.array([r[4:4+NUM_TX] for r in usrp_rows], dtype=np.complex64)       # (K, 3)
t_os_arr = np.array([r[0] for r in usrp_rows])
t_usrp_arr = np.array([r[1] for r in usrp_rows])
frame_arr = np.array([r[2] for r in usrp_rows], dtype=np.int32)
sym_arr = np.array([r[3] for r in usrp_rows], dtype=np.int32)

# 5 ms CSI (후배 형식 csi = (T, TX)): 같은 반 프레임의 세 심볼을 평균
win_key = frame_arr.astype(np.int64) * 2 + (sym_arr >= 70)
_, first_idx, counts = np.unique(win_key, return_index=True, return_counts=True)
full = counts == len(SYMS_PER_WINDOW)
csi = np.array([np.nanmean(H[i:i+c], axis=0) for i, c in zip(first_idx[full], counts[full])], dtype=np.complex64).reshape(-1, NUM_TX)
csi_t_usrp = np.array([t_usrp_arr[i:i+c].mean() for i, c in zip(first_idx[full], counts[full])])

np.savez_compressed(
    out_dir / "usrp_sparse_h.npz",
    H=H, t_os=t_os_arr, t_usrp=t_usrp_arr, frame_idx=frame_arr, sym_in_frame=sym_arr,
    csi=csi, csi_t_usrp=csi_t_usrp,
    sc_indices=sc_indices, sampling_rate=np.float64(sampling_rate), fs_actual=np.float64(fs_actual),
    frame_length=np.int64(frame_length), SYMS_PER_FRAME=np.int64(SYMS_PER_FRAME),
    carrier_frequency=np.float64(carrier_frequency), Rx_gain=np.float64(Rx_gain), Tx_gains=np.array(Tx_gains),
    BANDWIDTH=np.float64(BANDWIDTH), N=np.int64(N), FFT_SIZE=np.int64(FFT_SIZE), PILOT_SEED=np.int64(PILOT_SEED),
    overflow_count=np.int64(overflow_count), gap_log=np.array(gap_log, dtype=np.float64).reshape(-1, 2),
)

cfg = {
    "code": "DMIMO_3TX_Turtlebot_Rx.py (d_mimo_1017_3iso)",
    "RUN_SECONDS": RUN_SECONDS, "SCENARIO_ID": SCENARIO_ID, "SCENARIO_NAME": f"bounded_random_S{SCENARIO_ID:02d}",
    "ANT_TAG": ANT_TAG, "SCENARIO_DURATION_S": TOTAL_DURATION,
    "PRE_ROLL_SECONDS": PRE_ROLL_SECONDS, "POST_ROLL_SECONDS": POST_ROLL_SECONDS,
    "t0_wall": t0_wall, "t0_usrp_est": t0_usrp_est, "host_usrp_clock_map": clock_map,
    "WINDOW_SEC": WINDOW_SEC, "SYMS_PER_WINDOW": SYMS_PER_WINDOW,
    "carrier_frequency": carrier_frequency, "BANDWIDTH": BANDWIDTH, "N": N, "FFT_SIZE": FFT_SIZE,
    "sampling_rate": sampling_rate, "fs_actual": fs_actual,
    "frame_length_samples": frame_length, "frame_duration_sec": frame_duration_sec,
    "subcarriers": sc_list, "pilot_seed": PILOT_SEED,
    "tx": [{"name": TX_NAMES[i], "subcarrier": sc_list[i], "gain": Tx_gains[i], "antenna": TX_ANT} for i in range(NUM_TX)],
    "rx": {"addr": TRX_ADDR, "subdev": RX_SUBDEV, "antenna": RX_ANT, "gain": Rx_gain},
    "clock": {"mb0": [CLOCK_SRC0, TIME_SRC0], "mb1": [CLOCK_SRC1, TIME_SRC1], "ref_locked": ref_locked,
              "mb_time_diff_s": mb_time_diff},
    "tx_power_scale": POWER, "tx_lo_offset_hz": lo_offset,
    "sync_quality": sync_quality, "frame0_start_abs": frame0_start_abs,
    "ring_peak": ring_peak, "no_robot": NO_ROBOT,
    "overflow_count": overflow_count, "gaps_n": len(gap_log),
    "gaps_total_ms": sum(g for _, g in gap_log) / fs_actual * 1e3, "other_err_count": other_err_count,
    "tx_underflow": tx_state["underflow"], "tx_seq_err": tx_state["seq_err"], "tx_time_err": tx_state["time_err"],
    "tx_error": tx_state["error"],
}
with open(out_dir / "config.json", "w") as f:
    json.dump(cfg, f, indent=2, default=str)

# [YT 1019] 날짜 폴더 색인: logs/YYYYMMDD/index.csv 에 trial 한 줄씩 덧붙인다
try:
    import csv as _csv
    _med = []
    for i in range(NUM_TX):
        a_ = np.abs(H[:, i]); a_ = a_[np.isfinite(a_)]
        _med.append(f"{20*np.log10(np.median(a_)+1e-12):.1f}" if a_.size else "")
    _row = {"time": stamp, "folder": out_dir.name, "scenario": SCENARIO_ID, "tag": ANT_TAG, "no_robot": int(NO_ROBOT),
            "run_s": round(RUN_SECONDS, 1), "rx_gain": Rx_gain, "tx_gain": "/".join(f"{g:g}" for g in Tx_gains),
            "ref_locked": "/".join(str(v) for v in ref_locked), "mb_time_diff_ms": round(mb_time_diff * 1e3, 3),
            "sync": round(sync_quality, 1), "overflow": overflow_count, "gaps": len(gap_log),
            "tx_underflow": tx_state["underflow"], "peak": round(ring_peak, 3),
            "TX0_med_dB": _med[0], "TX1_med_dB": _med[1], "TX2_med_dB": _med[2], "live_udp": LIVE_UDP, "note": NOTE}
    _idx = SAVE_ROOT / "index.csv"
    _new = not _idx.exists()
    with open(_idx, "a", newline="", encoding="utf-8") as _f:
        _w = _csv.DictWriter(_f, fieldnames=list(_row))
        if _new:
            _w.writeheader()
        _w.writerow(_row)
    print(f"[INDEX] {_idx} 에 한 줄 추가")
except Exception as _e:  # noqa: BLE001 — 색인 실패가 측정 저장을 막지 않게
    print(f"[INDEX] 추가 실패: {_e}")

# 한눈 요약: TX 별 |H| (dB) — 셋 다 잡음 위로 올라와 있어야 한다
print("\n=== TX 별 |H| (전체 구간) ===")
for i in range(NUM_TX):
    a = np.abs(H[:, i]); a = a[np.isfinite(a)]
    if a.size:
        print(f"  {TX_NAMES[i]:22s} sc{sc_list[i]:<3d} 중앙 {20*np.log10(np.median(a)+1e-12):+6.1f} dB · "
              f"범위(5~95%) {20*np.log10(np.percentile(a,5)+1e-12):+6.1f} ~ {20*np.log10(np.percentile(a,95)+1e-12):+6.1f} dB")
print("  (−60 dB 근처면 그 TX 는 안 들어온 것: 케이블·포트·게인 확인)")

print("\n=== SAVED ===")
print("Dir      :", out_dir)
print("USRP CSV :", usrp_csv)
if scenario_usrp_csv is not None:
    print("40-s H   :", scenario_usrp_csv)


# =========================
# PLOT
# =========================
if os.environ.get("NO_PLOT", "0") != "1":
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    ph = np.unwrap(np.nan_to_num(np.angle(H)), axis=0)
    t_rel = t_usrp_arr - (t0_usrp_est if t0_usrp_est is not None else t_usrp_arr[0])
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i in range(NUM_TX):
        ax[0].plot(t_rel, mag_db[:, i], lw=0.6, label=f"TX{i} sc{sc_list[i]}")
        ax[1].plot(t_rel, ph[:, i], lw=0.6, label=f"TX{i}")
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        ax[2].plot(t_rel, np.unwrap(np.nan_to_num(np.angle(H[:, i] * np.conj(H[:, j])))), lw=0.6, label=f"TX{i}-TX{j}")
    ax[0].set_ylabel("|H| [dB]"); ax[1].set_ylabel("phase [rad]"); ax[2].set_ylabel("rel. phase [rad]")
    ax[2].set_xlabel("t - t0 [s]")
    for a in ax:
        a.legend(loc="best", ncol=3); a.grid(alpha=0.2)
        if t0_usrp_est is not None:
            a.axvline(0, color="w", lw=0.5, ls="--"); a.axvline(TOTAL_DURATION, color="w", lw=0.5, ls="--")
    ax[0].set_title(f"{EXP_TAG}  (3 TX ISO -> RX turtlebot)")
    fig.tight_layout()
    fig.savefig(out_dir / "plot.png", dpi=150)
    print("Plot PNG:", out_dir / "plot.png")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()

del tx_streamer, rx_streamer, usrp
print("Done.", flush=True)
import sys as _s; _s.stdout.flush(); _s.stderr.flush()
os._exit(0)                                                    # UHD/rclpy 스레드 정리 중 멈춤을 피한다
