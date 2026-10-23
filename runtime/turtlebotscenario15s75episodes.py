#!/usr/bin/env python3
"""
Forward-only, translation-dominant 15-s TurtleBot4 mobility episodes.

Design goals
------------
1) 75 episodes are independent deterministic pseudo-random routes.
2) Motion is dominated by forward translation, not in-place spinning.
3) No reverse command is ever used (linear.x >= 0 always).
4) Each route is a sequence of long straight-ish legs connected by short heading changes.
5) Nominal route waypoints remain inside a 0.52 m disk centered at the start pose.
   Because a disk is convex, the ideal straight line between any two waypoints also
   remains inside that disk.
6) A soft safety boundary at 0.58 m actively drives inward; 0.62 m aborts the trial.
7) The same scenario ID reproduces the same waypoint route across FAS/ISO trials.
8) Episode duration is exactly 15.0 s in host monotonic time.
9) Dataset roles are intentionally left unassigned during collection. After all 75
   episodes have been measured, build_dataset.py assigns 65 train, 5 validation,
   and 5 test episodes reproducibly.

Examples
--------
  python3 turtle_scenarios_15s_75episodes.py --list
  python3 turtle_scenarios_15s_75episodes.py --scenario 1 --dry-run
  python3 turtle_scenarios_15s_75episodes.py --scenario 71

This module preserves the same SCENARIOS / TOTAL_DURATION / run_scenario API used by
FAS_2TX_Turtlebot_Rx_t0.py, so the RX integration code does not need to change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped

from .turtle_control import TB4AdvancedControl


# =============================================================================
# Experiment parameters
# =============================================================================
TOTAL_DURATION = 15.0
CONTROL_HZ = 20.0
MASTER_SEED = 20261004
WAIT_POSE_TIMEOUT_SEC = 30.0

# Each generated scenario is one 15-s episode. Dataset roles are assigned only
# after collection, by build_dataset.py.
NUM_EPISODES = 75

# Geometry. The nominal route is deliberately smaller than 0.60 m to leave tracking
# margin.  The robot should almost always remain inside the requested 60-cm circle.
WAYPOINT_RADIUS_MAX_M = 0.52
WAYPOINT_RADIUS_MIN_M = 0.30
SOFT_RADIUS_M = 0.58
HARD_RADIUS_M = 0.62

# Route generation: large spatial displacement between successive targets.
N_WAYPOINTS = 16
MIN_LEG_LENGTH_M = 0.34
PREFERRED_LEG_LENGTH_M = 0.48

# Controller: faster translation and short turns.  No reverse velocity is permitted.
V_CRUISE_MIN = 0.105
V_CRUISE_MAX = 0.145
V_APPROACH_MIN = 0.055
W_ALIGN = 1.00                 # fast in-place heading alignment [rad/s]
W_DRIVE_MAX = 0.42             # only mild correction while translating
K_W_DRIVE = 1.10
ALIGN_ENTER_DEG = 32.0         # align in place only for a large heading error
ALIGN_EXIT_DEG = 8.0           # resume translation as soon as alignment is good
WAYPOINT_TOL_M = 0.055


@dataclass(frozen=True)
class Waypoint:
    x: float                   # local x [m] relative to scenario start pose
    y: float                   # local y [m]
    speed: float               # forward cruise speed [m/s]


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    spawn_key: Tuple[int, ...]
    waypoints: Tuple[Waypoint, ...]
    nominal_path_length_m: float


def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def dataset_split(scenario_id: int) -> str:
    """Collection-time label; build_dataset.py assigns the final split later."""
    if not 1 <= scenario_id <= NUM_EPISODES:
        raise ValueError(f"scenario_id must be in 1..{NUM_EPISODES}: {scenario_id}")
    return "unassigned"


# =============================================================================
# Scenario generation
# =============================================================================
def _generate_waypoint_route(rng: np.random.Generator) -> Tuple[Waypoint, ...]:
    """
    Generate a forward polyline inside the bounded disk.

    The route is built sequentially from the previous travel direction.  Each new leg
    is long (roughly 0.32--0.56 m) and the change of heading is usually 25--105 deg.
    This produces the requested pattern: straight translation -> short turn -> straight
    translation, instead of repeated in-place rotation.
    """
    points = []

    # First leg: move substantially away from the start point.
    heading = float(rng.uniform(-math.pi, math.pi))
    first_len = float(rng.uniform(0.38, 0.50))
    x = first_len * math.cos(heading)
    y = first_len * math.sin(heading)
    points.append(Waypoint(x, y, float(rng.uniform(V_CRUISE_MIN, V_CRUISE_MAX))))

    prev_x, prev_y = x, y

    for _ in range(1, N_WAYPOINTS):
        accepted = None

        # Prefer moderate turns, with a random CW/CCW direction and long translation.
        for _attempt in range(1800):
            sign = -1.0 if float(rng.uniform()) < 0.5 else 1.0
            delta = sign * math.radians(float(rng.uniform(22.0, 92.0)))
            cand_heading = wrap_pi(heading + delta)
            leg = float(rng.uniform(0.32, 0.56))

            cx = prev_x + leg * math.cos(cand_heading)
            cy = prev_y + leg * math.sin(cand_heading)
            cr = math.hypot(cx, cy)

            if cr > WAYPOINT_RADIUS_MAX_M:
                continue

            # Avoid tiny center dithering and nearly overlapping vertices.
            if math.hypot(cx - prev_x, cy - prev_y) < MIN_LEG_LENGTH_M:
                continue

            accepted = (cx, cy, cand_heading)
            break

        if accepted is None:
            # Boundary fallback: search broader but still cap the heading change so
            # a single waypoint never requires a near-180-degree spin.
            best = None
            best_turn = None
            for sign in (-1.0, 1.0):
                for turn_deg in np.linspace(55.0, 120.0, 27):
                    cand_heading = wrap_pi(heading + sign * math.radians(float(turn_deg)))
                    for leg in np.linspace(0.30, 0.52, 23)[::-1]:
                        cx = prev_x + float(leg) * math.cos(cand_heading)
                        cy = prev_y + float(leg) * math.sin(cand_heading)
                        if math.hypot(cx, cy) <= WAYPOINT_RADIUS_MAX_M:
                            # Prefer the smallest feasible turn and a long leg.
                            score = float(turn_deg) - 25.0 * float(leg)
                            if best is None or score < best_turn:
                                best = (cx, cy, cand_heading)
                                best_turn = score
                            break
            if best is None:
                raise RuntimeError("Could not generate bounded forward leg")
            accepted = best

        prev_x, prev_y, heading = accepted
        speed = float(rng.uniform(V_CRUISE_MIN, V_CRUISE_MAX))
        points.append(Waypoint(prev_x, prev_y, speed))

    return tuple(points)


def _route_length(waypoints: Tuple[Waypoint, ...]) -> float:
    x0, y0 = 0.0, 0.0
    s = 0.0
    for w in waypoints:
        s += math.hypot(w.x - x0, w.y - y0)
        x0, y0 = w.x, w.y
    return s


def _generate_one_scenario(scenario_id: int, seed_seq: np.random.SeedSequence) -> Scenario:
    rng = np.random.default_rng(seed_seq)

    # Re-generate until the nominal path is sufficiently long that a 15-s run cannot
    # trivially finish all waypoints.  This also guarantees translation-rich motion.
    for _ in range(200):
        waypoints = _generate_waypoint_route(rng)
        length = _route_length(waypoints)
        max_r = max(math.hypot(w.x, w.y) for w in waypoints)
        min_leg = min(
            math.hypot(
                waypoints[i].x - (0.0 if i == 0 else waypoints[i-1].x),
                waypoints[i].y - (0.0 if i == 0 else waypoints[i-1].y),
            )
            for i in range(len(waypoints))
        )
        if length >= 5.4 and max_r <= WAYPOINT_RADIUS_MAX_M + 1e-9 and min_leg >= 0.30:
            return Scenario(
                scenario_id=scenario_id,
                spawn_key=tuple(seed_seq.spawn_key),
                waypoints=waypoints,
                nominal_path_length_m=length,
            )

    raise RuntimeError(f"Failed to generate a valid route for scenario {scenario_id}")


def build_scenarios() -> dict[int, Scenario]:
    master = np.random.SeedSequence(MASTER_SEED)
    children = master.spawn(NUM_EPISODES)
    return {
        sid: _generate_one_scenario(sid, children[sid - 1])
        for sid in range(1, NUM_EPISODES + 1)
    }


SCENARIOS = build_scenarios()


# =============================================================================
# Geometry / ROS helpers
# =============================================================================
def local_to_map(x_local: float, y_local: float,
                 start_x: float, start_y: float, start_yaw_deg: float) -> Tuple[float, float]:
    yaw = math.radians(start_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return (
        start_x + c * x_local - s * y_local,
        start_y + s * x_local + c * y_local,
    )


def publish_velocity(node: TB4AdvancedControl, linear_x: float, angular_z: float) -> None:
    # Explicitly prohibit reverse commands.
    linear_x = max(0.0, float(linear_x))
    msg = TwistStamped()
    msg.header.frame_id = "base_link"
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.twist.linear.x = linear_x
    msg.twist.angular.z = float(angular_z)
    node.pub.publish(msg)


def stop_robot(node: TB4AdvancedControl) -> None:
    for _ in range(5):
        publish_velocity(node, 0.0, 0.0)
        time.sleep(0.04)


def wait_for_pose(node: TB4AdvancedControl, timeout: float = WAIT_POSE_TIMEOUT_SEC):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pose = node.get_pose()
        if pose is not None:
            return pose
        time.sleep(0.1)
    raise RuntimeError("No TurtleBot pose received. Check SLAM / ROS_DOMAIN_ID / network.")


def _target_heading_and_distance(pose, target_map: Tuple[float, float]):
    x, y, yaw_deg = pose
    dx = target_map[0] - x
    dy = target_map[1] - y
    dist = math.hypot(dx, dy)
    desired = math.atan2(dy, dx)
    err = wrap_pi(desired - math.radians(yaw_deg))
    return dist, err


def _forward_drive_controller(pose, target_map: Tuple[float, float], cruise_v: float,
                              align_mode: bool) -> Tuple[float, float, float, float, bool, str]:
    """
    Translation-dominant waypoint controller.

    - Large heading error: brief in-place ALIGN at high angular rate.
    - Small/moderate error: positive forward velocity with only mild steering correction.
    - No reverse motion under any condition.
    """
    dist, err = _target_heading_and_distance(pose, target_map)
    abs_err = abs(err)

    if dist <= WAYPOINT_TOL_M:
        return 0.0, 0.0, dist, err, False, "REACHED"

    # Hysteresis prevents chatter between ALIGN and DRIVE.
    if align_mode:
        if abs_err <= math.radians(ALIGN_EXIT_DEG):
            align_mode = False
    elif abs_err >= math.radians(ALIGN_ENTER_DEG):
        align_mode = True

    if align_mode:
        w = math.copysign(W_ALIGN, err)
        return 0.0, w, dist, err, True, "ALIGN"

    # Forward translation. Reduce speed only near waypoint or with residual heading error.
    heading_scale = max(0.58, math.cos(err))
    distance_scale = min(1.0, max(0.45, dist / 0.22))
    v = max(V_APPROACH_MIN, cruise_v * heading_scale * distance_scale)
    w = max(-W_DRIVE_MAX, min(W_DRIVE_MAX, K_W_DRIVE * err))
    return v, w, dist, err, False, "DRIVE"


# =============================================================================
# Logging / scenario definition
# =============================================================================
def scenario_definition_dict(s: Scenario) -> dict:
    return {
        "scenario_id": s.scenario_id,
        "episode_id": s.scenario_id,
        "dataset_split": dataset_split(s.scenario_id),
        "dataset_split_stage": "assigned_after_collection_by_build_dataset",
        "num_episodes_total": NUM_EPISODES,
        "master_seed": MASTER_SEED,
        "spawn_key": list(s.spawn_key),
        "duration_s": TOTAL_DURATION,
        "route_type": "forward_only_translation_dominant_waypoints",
        "waypoint_radius_max_m": WAYPOINT_RADIUS_MAX_M,
        "soft_radius_m": SOFT_RADIUS_M,
        "hard_radius_m": HARD_RADIUS_M,
        "nominal_path_length_m": s.nominal_path_length_m,
        "waypoints": [asdict(w) for w in s.waypoints],
    }


def _scenario_stats(s: Scenario) -> dict:
    max_r = max(math.hypot(w.x, w.y) for w in s.waypoints)
    legs = []
    prev = (0.0, 0.0)
    for w in s.waypoints:
        legs.append(math.hypot(w.x - prev[0], w.y - prev[1]))
        prev = (w.x, w.y)
    return {
        "max_waypoint_radius": max_r,
        "min_leg": min(legs),
        "mean_leg": sum(legs) / len(legs),
        "max_leg": max(legs),
        "path_length": sum(legs),
    }


def print_scenario(s: Scenario) -> None:
    stats = _scenario_stats(s)
    print(
        f"\nEpisode {s.scenario_id} [{dataset_split(s.scenario_id)}] | "
        f"spawn_key={s.spawn_key} | "
        f"path={stats['path_length']:.2f} m | max_r={stats['max_waypoint_radius']:.2f} m"
    )
    print(
        f"Leg length: min={stats['min_leg']:.2f}, mean={stats['mean_leg']:.2f}, "
        f"max={stats['max_leg']:.2f} m"
    )
    print(" idx      x[m]      y[m]      r[m]    speed[m/s]    leg[m]")
    print(" ----  --------  --------  --------  -----------  --------")
    prev = (0.0, 0.0)
    for i, w in enumerate(s.waypoints):
        r = math.hypot(w.x, w.y)
        leg = math.hypot(w.x - prev[0], w.y - prev[1])
        print(f" {i:>3d}   {w.x:+.3f}    {w.y:+.3f}     {r:.3f}       {w.speed:.3f}       {leg:.3f}")
        prev = (w.x, w.y)


# =============================================================================
# Scenario runner -- API kept compatible with previous version
# =============================================================================
def run_scenario(node: TB4AdvancedControl,
                 scenario_id: int,
                 out_dir: Path,
                 countdown: float = 3.0,
                 start_callback=None,
                 rate_hz: float = CONTROL_HZ) -> dict:
    scenario = SCENARIOS[scenario_id]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "scenario_definition.json", "w") as f:
        json.dump(scenario_definition_dict(scenario), f, indent=2)

    start_pose = wait_for_pose(node)
    sx, sy, syaw = start_pose

    print(
        f"[READY] Episode {scenario_id} [{dataset_split(scenario_id)}]: "
        "forward-only waypoint route"
    )
    print(f"[READY] Start pose = ({sx:.3f}, {sy:.3f}), yaw={syaw:.2f} deg")
    print(
        f"[READY] nominal path={scenario.nominal_path_length_m:.2f} m, "
        f"waypoints r<={WAYPOINT_RADIUS_MAX_M:.2f} m"
    )
    print("[READY] linear.x is constrained to >= 0 at all times")

    if countdown > 0:
        deadline = time.monotonic() + countdown
        print(f"[READY] Starting in {countdown:.1f} s ...")
        while time.monotonic() < deadline:
            publish_velocity(node, 0.0, 0.0)
            time.sleep(0.05)

    # Marker immediately before the first route-control command.
    t0_wall = time.time()
    t0_mono = time.monotonic()
    callback_data = start_callback(t0_wall, t0_mono) if start_callback else None

    marker = {
        "scenario_id": scenario_id,
        "episode_id": scenario_id,
        "dataset_split": dataset_split(scenario_id),
        "t0_wall": t0_wall,
        "t0_monotonic": t0_mono,
        "duration_s": TOTAL_DURATION,
        "start_pose": {"x": sx, "y": sy, "yaw_deg": syaw},
        "callback_data": callback_data,
    }
    with open(out_dir / "scenario_event.json", "w") as f:
        json.dump(marker, f, indent=2)

    rows = []
    waypoint_idx = 0
    align_mode = True
    reached_count = 0
    aborted = False
    abort_reason: Optional[str] = None
    period = 1.0 / float(rate_hz)
    next_tick = t0_mono
    drive_time = 0.0
    align_time = 0.0
    recover_time = 0.0
    prev_mode_used: Optional[str] = None

    print(f"[T0] Episode {scenario_id} started at host wall time {t0_wall:.6f}")

    try:
        while True:
            now = time.monotonic()
            t_rel = now - t0_mono
            if t_rel >= TOTAL_DURATION:
                break

            pose = node.get_pose()
            if pose is None:
                publish_velocity(node, 0.0, 0.0)
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
                continue

            x, y, yaw_deg = pose
            radius = math.hypot(x - sx, y - sy)

            if radius >= HARD_RADIUS_M:
                aborted = True
                abort_reason = f"hard_radius_exceeded:{radius:.3f}m"
                print(f"[SAFETY] Hard radius exceeded ({radius:.3f} m). Aborting.")
                break

            # Nominal route target in the map frame.
            wp = scenario.waypoints[waypoint_idx % len(scenario.waypoints)]
            target_map = local_to_map(wp.x, wp.y, sx, sy, syaw)

            if radius >= SOFT_RADIUS_M:
                # Drive toward the start region using forward-only control.
                recovery_target = (sx, sy)
                v_cmd, w_cmd, dist, heading_err, align_mode, inner_mode = _forward_drive_controller(
                    pose, recovery_target, cruise_v=0.10, align_mode=align_mode
                )
                target_used = recovery_target
                mode_used = "RECOVER_ALIGN" if inner_mode == "ALIGN" else "RECOVER_DRIVE"
            else:
                v_cmd, w_cmd, dist, heading_err, align_mode, inner_mode = _forward_drive_controller(
                    pose, target_map, cruise_v=wp.speed, align_mode=align_mode
                )
                target_used = target_map
                mode_used = inner_mode

                if inner_mode == "REACHED":
                    reached_count += 1
                    waypoint_idx += 1
                    align_mode = True
                    wp = scenario.waypoints[waypoint_idx % len(scenario.waypoints)]
                    target_map = local_to_map(wp.x, wp.y, sx, sy, syaw)
                    # Immediately start aligning toward the next long straight leg.
                    v_cmd, w_cmd, dist, heading_err, align_mode, inner_mode = _forward_drive_controller(
                        pose, target_map, cruise_v=wp.speed, align_mode=True
                    )
                    target_used = target_map
                    mode_used = inner_mode

            publish_velocity(node, v_cmd, w_cmd)

            if prev_mode_used is not None:
                if "RECOVER" in prev_mode_used:
                    recover_time += period
                elif prev_mode_used == "DRIVE":
                    drive_time += period
                elif "ALIGN" in prev_mode_used:
                    align_time += period
            prev_mode_used = mode_used

            rows.append({
                "t_rel": t_rel,
                "t_wall": time.time(),
                "waypoint_idx": waypoint_idx,
                "mode_used": mode_used,
                "x": x,
                "y": y,
                "yaw_deg": yaw_deg,
                "radius_from_start_m": radius,
                "target_map_x": target_used[0],
                "target_map_y": target_used[1],
                "target_local_x": wp.x,
                "target_local_y": wp.y,
                "distance_to_target_m": dist,
                "heading_error_rad": heading_err,
                "cmd_linear_x": max(0.0, v_cmd),
                "cmd_angular_z": w_cmd,
            })

            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))

    except KeyboardInterrupt:
        aborted = True
        abort_reason = "keyboard_interrupt"
        print("\n[INFO] Interrupted by user.")
    finally:
        stop_robot(node)

    t_end_wall = time.time()
    t_end_mono = time.monotonic()

    csv_path = out_dir / "turtle_trajectory.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    max_radius_actual = max((r["radius_from_start_m"] for r in rows), default=float("nan"))
    forward_distance_cmd = sum(
        r["cmd_linear_x"] * period for r in rows if math.isfinite(r["cmd_linear_x"])
    )

    summary = {
        **marker,
        "t_end_wall": t_end_wall,
        "t_end_monotonic": t_end_mono,
        "actual_elapsed_s": t_end_mono - t0_mono,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "num_logged_points": len(rows),
        "waypoints_reached": reached_count,
        "max_radius_actual_m": max_radius_actual,
        "approx_forward_command_distance_m": forward_distance_cmd,
        "drive_time_s_approx": drive_time,
        "align_time_s_approx": align_time,
        "recover_time_s_approx": recover_time,
        "trajectory_csv": str(csv_path),
    }
    with open(out_dir / "scenario_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[DONE] elapsed={summary['actual_elapsed_s']:.3f} s, "
        f"reached={reached_count}, max_r={max_radius_actual:.3f} m, aborted={aborted}"
    )
    print(
        f"[MOTION] forward_cmd_dist~{forward_distance_cmd:.2f} m, "
        f"drive~{drive_time:.1f}s, align~{align_time:.1f}s"
    )
    print(f"[SAVE] {csv_path}")
    return summary


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=int,
        choices=range(1, NUM_EPISODES + 1),
        help=f"run one episode/scenario (1..{NUM_EPISODES})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help=f"show summaries for all {NUM_EPISODES} unassigned episodes",
    )
    parser.add_argument("--dry-run", action="store_true", help="print route only; do not connect to ROS")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("./scenario_logs"))
    args = parser.parse_args()

    if args.list:
        scenario_ids = range(1, NUM_EPISODES + 1)
        print("Dataset split: unassigned during collection; run build_dataset.py after all episodes")
        for sid in scenario_ids:
            s = SCENARIOS[sid]
            st = _scenario_stats(s)
            print(
                f"E{sid:02d} [{dataset_split(sid):>10s}]: "
                f"path={st['path_length']:.2f}m, "
                f"leg mean={st['mean_leg']:.2f}m, min={st['min_leg']:.2f}m, "
                f"max_r={st['max_waypoint_radius']:.2f}m, spawn_key={s.spawn_key}"
            )
        return

    if args.scenario is None:
        parser.error(f"Use --scenario 1..{NUM_EPISODES}, or --list")

    print_scenario(SCENARIOS[args.scenario])
    if args.dry_run:
        return

    rclpy.init()
    node = TB4AdvancedControl(f"forward_episode_{args.scenario:02d}")
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    try:
        split_name = dataset_split(args.scenario)
        run_dir = (
            args.out
            / split_name
            / f"episode_{args.scenario:02d}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        run_scenario(node, args.scenario, run_dir, countdown=args.countdown)
    finally:
        try:
            stop_robot(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
