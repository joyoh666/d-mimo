# -*- coding: utf-8 -*-
"""
hy_robot_proc.py — 터틀봇 제어만 따로 도는 자식 프로세스   2026-10-04
FAS_2TX_Turtlebot_Rx_t0.py 가 ROBOT_PROC=1(기본) 일 때 스스로 띄운다. 손으로 실행할 일은 없다.

왜 따로 — USRP 수신과 한 파이썬 프로세스에서 돌면 로봇 위치 갱신이 20 -> 5~6 Hz 로 떨어지고
경유점 7 -> 4 -> 2 로 경로를 못 탔다 (10-04 TEST7·TEST8 실측). 프로세스를 나누면 CPU 를 따로 쓴다.

부모와의 약속 (stdout 한 줄씩, 앞에 @@):
  @@HOME <x> <y> <yaw> <file> 최초 배치에서 저장한 SLAM map pose
  @@READY                   로봇 위치를 받기 시작함
  (부모가 stdin 에 GO)
  @@T0 <t0_wall> <t0_mono>  선배 run_scenario 의 t0 (첫 /cmd_vel 직전). monotonic 은 같은 PC 라 부모와 공유된다
  @@DONE <json>             시나리오 실행 및 최초 pose 복귀가 모두 성공한 뒤의 요약
  @@ERROR <메시지>
그 밖의 줄([READY] [T0] [DONE] …)은 그대로 부모 콘솔에 찍힌다.
로봇 상태 로그(선배 turtlebot_states.csv 의 원자료)는 <out>/robot_states_raw.csv 로 남긴다.
"""
import csv
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import rclpy

from .turtle_control import TB4AdvancedControl
from .turtlebotscenario15s75episodes import publish_velocity, run_scenario, stop_robot


def say(msg):
    print(msg, flush=True)


def _wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _load_or_create_home_pose(current_pose):
    """배치 최초의 SLAM map pose를 저장하고 이후 시나리오에서 그대로 재사용한다."""
    pose_file_text = os.environ.get("HOME_POSE_FILE", "").strip()
    if not pose_file_text:
        return tuple(float(v) for v in current_pose), None

    pose_file = Path(pose_file_text)
    if pose_file.exists():
        with open(pose_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        pose = (float(data["x"]), float(data["y"]), float(data["yaw_deg"]))
        return pose, pose_file

    pose_file.parent.mkdir(parents=True, exist_ok=True)
    pose = tuple(float(v) for v in current_pose)
    with open(pose_file, "w", encoding="utf-8") as f:
        json.dump(
            {"x": pose[0], "y": pose[1], "yaw_deg": pose[2], "captured_at": time.time()},
            f,
            indent=2,
        )
    return pose, pose_file


def return_to_home(node, home_pose):
    """SLAM pose 피드백으로 최초 위치와 방향까지 복귀한다."""
    timeout = float(os.environ.get("RETURN_TIMEOUT_SEC", "35.0"))
    position_tol = float(os.environ.get("RETURN_POSITION_TOL_M", "0.05"))
    yaw_tol_deg = float(os.environ.get("RETURN_YAW_TOL_DEG", "5.0"))
    control_hz = float(os.environ.get("RETURN_CONTROL_HZ", "15.0"))
    max_linear = float(os.environ.get("RETURN_MAX_LINEAR_MPS", "0.12"))
    max_angular = float(os.environ.get("RETURN_MAX_ANGULAR_RPS", "0.80"))

    hx, hy, hyaw_deg = home_pose
    deadline = time.monotonic() + timeout
    period = 1.0 / control_hz
    stable_count = 0
    missing_pose_since = None
    last_report = 0.0
    last_pose = None

    say(
        f"[RETURN] 최초 pose로 복귀 시작: "
        f"x={hx:.3f}, y={hy:.3f}, yaw={hyaw_deg:.2f} deg"
    )
    try:
        while time.monotonic() < deadline:
            loop_start = time.monotonic()
            pose = node.get_pose()
            if pose is None:
                publish_velocity(node, 0.0, 0.0)
                if missing_pose_since is None:
                    missing_pose_since = loop_start
                elif loop_start - missing_pose_since > 2.0:
                    return {
                        "success": False,
                        "reason": "slam_pose_lost",
                        "target_pose": {"x": hx, "y": hy, "yaw_deg": hyaw_deg},
                    }
                time.sleep(period)
                continue

            missing_pose_since = None
            last_pose = pose
            x, y, yaw_deg = pose
            dx, dy = hx - x, hy - y
            distance = math.hypot(dx, dy)
            final_yaw_error = _wrap_pi(math.radians(hyaw_deg - yaw_deg))

            if distance > position_tol:
                target_heading = math.atan2(dy, dx)
                heading_error = _wrap_pi(target_heading - math.radians(yaw_deg))
                if abs(heading_error) > math.radians(22.0):
                    linear = 0.0
                    angular = max(-max_angular, min(max_angular, 1.8 * heading_error))
                else:
                    linear = min(max_linear, max(0.035, 0.65 * distance))
                    linear *= max(0.35, math.cos(heading_error))
                    angular = max(-0.55, min(0.55, 1.5 * heading_error))
                stable_count = 0
                phase = "position"
            elif abs(math.degrees(final_yaw_error)) > yaw_tol_deg:
                linear = 0.0
                angular = max(-max_angular, min(max_angular, 1.8 * final_yaw_error))
                stable_count = 0
                phase = "yaw"
            else:
                linear = angular = 0.0
                stable_count += 1
                phase = "settling"

            publish_velocity(node, linear, angular)
            if loop_start - last_report >= 1.0:
                say(
                    f"[RETURN] {phase}: distance={distance:.3f} m, "
                    f"yaw_error={math.degrees(final_yaw_error):+.1f} deg"
                )
                last_report = loop_start

            # 순간적으로 허용 오차 안에 들어온 경우가 아니라 5회 연속 확인한다.
            if stable_count >= 5:
                stop_robot(node)
                return {
                    "success": True,
                    "reason": None,
                    "target_pose": {"x": hx, "y": hy, "yaw_deg": hyaw_deg},
                    "final_pose": {"x": x, "y": y, "yaw_deg": yaw_deg},
                    "position_error_m": distance,
                    "yaw_error_deg": math.degrees(final_yaw_error),
                }

            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
    finally:
        stop_robot(node)

    result = {
        "success": False,
        "reason": "return_timeout",
        "target_pose": {"x": hx, "y": hy, "yaw_deg": hyaw_deg},
    }
    if last_pose is not None:
        x, y, yaw_deg = last_pose
        result["final_pose"] = {"x": x, "y": y, "yaw_deg": yaw_deg}
        result["position_error_m"] = math.hypot(hx - x, hy - y)
        result["yaw_error_deg"] = math.degrees(_wrap_pi(math.radians(hyaw_deg - yaw_deg)))
    return result


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
    current_pose = None
    while current_pose is None:
        current_pose = node.get_pose()
        if current_pose is not None:
            break
        if time.time() - t > wait_s:
            say("@@ERROR Pose를 못 받았습니다. slam.launch.py 상태/ROS_DOMAIN_ID/네트워크 확인")
            return 2
        time.sleep(0.1)
    home_pose, home_pose_file = _load_or_create_home_pose(current_pose)
    say(
        f"@@HOME {home_pose[0]!r} {home_pose[1]!r} {home_pose[2]!r} "
        f"{'' if home_pose_file is None else str(home_pose_file)}"
    )
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
        if os.environ.get("RETURN_HOME", "1") == "1":
            return_result = return_to_home(node, home_pose)
            ev["return_home"] = return_result
            if not return_result["success"]:
                raise RuntimeError(
                    f"최초 위치 복귀 실패: {return_result['reason']} "
                    f"(result={return_result})"
                )
            say(
                f"[RETURN] 복귀 완료: position_error={return_result['position_error_m']:.3f} m, "
                f"yaw_error={return_result['yaw_error_deg']:+.1f} deg"
            )
        if ev.get("aborted"):
            raise RuntimeError(f"시나리오가 중단됨: {ev.get('abort_reason')}")
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
