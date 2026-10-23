# -*- coding: utf-8 -*-
"""
hy_robot_proc.py — 터틀봇 제어만 따로 도는 자식 프로세스   2026-10-04
FAS_2TX_Turtlebot_Rx_t0.py 가 ROBOT_PROC=1(기본) 일 때 스스로 띄운다. 손으로 실행할 일은 없다.

왜 따로 — USRP 수신과 한 파이썬 프로세스에서 돌면 로봇 위치 갱신이 20 -> 5~6 Hz 로 떨어지고
경유점 7 -> 4 -> 2 로 경로를 못 탔다 (10-04 TEST7·TEST8 실측). 프로세스를 나누면 CPU 를 따로 쓴다.

부모와의 약속 (stdout 한 줄씩, 앞에 @@):
  @@READY                   로봇 위치를 받기 시작함
  (부모가 stdin 에 GO)
  @@T0 <t0_wall> <t0_mono>  선배 run_scenario 의 t0 (첫 /cmd_vel 직전). monotonic 은 같은 PC 라 부모와 공유된다
  @@DONE <json>             run_scenario 의 요약
  @@ERROR <메시지>
그 밖의 줄([READY] [T0] [DONE] …)은 그대로 부모 콘솔에 찍힌다.
로봇 상태 로그(선배 turtlebot_states.csv 의 원자료)는 <out>/robot_states_raw.csv 로 남긴다.
"""
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import rclpy

from .turtle_control import TB4AdvancedControl
from .turtlebotscenario15s75episodes import run_scenario


def say(msg):
    print(msg, flush=True)


def main():
    sid = int(sys.argv[1]); out = Path(sys.argv[2]); pre = float(sys.argv[3])
    log_hz = float(os.environ.get("ROBOT_LOG_HZ", "5")); lag = float(os.environ.get("POSE_QUERY_LAG_SEC", "0.10"))
    wait_s = float(os.environ.get("WAIT_POSE_TIMEOUT_SEC", "30"))
    out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = TB4AdvancedControl("robot_proc_node")
    # 혼자 주행 스크립트(turtle_scenarios --scenario)와 같은 rclpy.spin — 그쪽은 위치 19.9 Hz.
    # 선배 통합 코드의 executor.spin_once(0.1) 반복은 6~7 Hz 였다 (10-04 TEST7~9 실측). TF 메시지를 못 따라가는 것으로 본다.
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    spin_stop = threading.Event()

    t = time.time()
    while node.get_pose() is None:
        if time.time() - t > wait_s:
            say("@@ERROR Pose를 못 받았습니다. slam.launch.py 상태/ROS_DOMAIN_ID/네트워크 확인")
            return 2
        time.sleep(0.1)
    say("@@READY")

    if sys.stdin.readline().strip() != "GO":
        say("@@ERROR 부모가 GO 를 안 보냈다")
        return 3

    rows, stop = [], threading.Event()

    def logger():                                            # 선배 robot_logger_loop 와 같은 내용
        dt, prev = 1.0 / log_hz, None
        while not stop.is_set():
            tp = time.time() - lag
            try:
                p = node.get_pose(tp)
            except TypeError:
                p = None
            if p is None:
                p = node.get_pose()
            if p is None:
                time.sleep(dt); continue
            x, y, z = p
            if prev is None:
                vx = vy = vz = 0.0
            else:
                t0, x0, y0, z0 = prev; d = max(1e-6, tp - t0)
                vx, vy, vz = (x - x0) / d, (y - y0) / d, (z - z0) / d
            prev = (tp, x, y, z)
            rows.append((tp, x, y, z, vx, vy, vz))
            time.sleep(dt)
    lt = threading.Thread(target=logger, daemon=True); lt.start()

    def cb(t0_wall, t0_mono):
        say(f"@@T0 {t0_wall!r} {t0_mono!r}")
        return {"t0_wall": t0_wall, "t0_monotonic": t0_mono}

    try:
        ev = run_scenario(node, sid, out, rate_hz=20.0, countdown=pre, start_callback=cb)
    except Exception as e:                                   # noqa: BLE001
        say(f"@@ERROR {e!r}")
        ev = None
    finally:
        stop.set(); lt.join(timeout=1.0)
        with open(out / "robot_states_raw.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_os", "x", "y", "z", "vx", "vy", "vz"]); w.writerows(rows)
        try:
            node.stop()
        except Exception:                                    # noqa: BLE001
            pass
    if ev is not None:
        say("@@DONE " + json.dumps(ev, default=str))
    spin_stop.set(); time.sleep(0.2)
    try:
        node.destroy_node(); rclpy.shutdown()
    except Exception:                                        # noqa: BLE001
        pass
    os._exit(0)                                              # rclpy 스레드 정리 중 core dump 를 피한다


if __name__ == "__main__":
    sys.exit(main())
