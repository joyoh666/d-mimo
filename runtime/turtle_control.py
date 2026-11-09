'''
ros2 launch turtlebot4_bringup lite.launch.py

이 코드 실행 전 turtlebot Raspberry Pi 접속하셔서 위 코드 실행하세요
'''

#%%
import os
import rclpy
from action_msgs.msg import GoalStatus
from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import TwistStamped
from tf2_ros import TransformListener, Buffer
import time, threading
import math

class TB4AdvancedControl(Node):
    def __init__(self, name='tb4_advanced_control', pose_frame=None):
        super().__init__(name)
        # Create 3 odometry is available without SLAM.  A different TF frame can
        # still be selected explicitly for backwards-compatible experiments.
        self.pose_frame = pose_frame or os.environ.get('POSE_FRAME', 'odom')
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._dock_status = None
        self._dock_status_received_at = None
        self._dock_status_sub = self.create_subscription(
            DockStatus,
            '/dock_status',
            self._dock_status_callback,
            qos_profile_sensor_data,
        )
        self._dock_client = ActionClient(self, Dock, '/dock')
        self._undock_client = ActionClient(self, Undock, '/undock')

    def _dock_status_callback(self, msg):
        self._dock_status = msg
        self._dock_status_received_at = time.monotonic()

    def dock_status(self):
        """Return the latest Create 3 dock sensor state, if one was received."""
        msg = self._dock_status
        if msg is None:
            return None
        return {
            'is_docked': bool(msg.is_docked),
            'dock_visible': bool(getattr(msg, 'dock_visible', False)),
            'received_at_monotonic': self._dock_status_received_at,
        }

    def wait_for_dock_status(self, timeout=10.0):
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            status = self.dock_status()
            if status is not None:
                return status
            time.sleep(0.05)
        return None

    @staticmethod
    def _wait_future(future, deadline):
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        return future.done()

    def _run_dock_action(self, action_name, client, goal, expected_docked, timeout):
        """Run a Create 3 dock/undock action and verify the dock sensor state."""
        started = time.monotonic()
        deadline = started + float(timeout)

        try:
            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                if client.wait_for_server(timeout_sec=min(0.5, remaining)):
                    break
            else:
                return {
                    'success': False,
                    'action': action_name,
                    'reason': 'action_server_unavailable',
                    'elapsed_s': time.monotonic() - started,
                    'dock_status': self.dock_status(),
                }
        except Exception as exc:  # noqa: BLE001 - convert ROS errors to batch results
            return {
                'success': False,
                'action': action_name,
                'reason': f'action_server_error:{exc!r}',
                'elapsed_s': time.monotonic() - started,
                'dock_status': self.dock_status(),
            }

        try:
            send_future = client.send_goal_async(goal)
        except Exception as exc:  # noqa: BLE001 - convert ROS errors to batch results
            return {
                'success': False,
                'action': action_name,
                'reason': f'goal_send_error:{exc!r}',
                'elapsed_s': time.monotonic() - started,
                'dock_status': self.dock_status(),
            }
        if not self._wait_future(send_future, deadline):
            return {
                'success': False,
                'action': action_name,
                'reason': 'goal_response_timeout',
                'elapsed_s': time.monotonic() - started,
                'dock_status': self.dock_status(),
            }

        try:
            goal_handle = send_future.result()
        except Exception as exc:  # noqa: BLE001 - report ROS action failure to parent
            return {
                'success': False,
                'action': action_name,
                'reason': f'goal_response_error:{exc!r}',
                'elapsed_s': time.monotonic() - started,
                'dock_status': self.dock_status(),
            }
        if goal_handle is None or not goal_handle.accepted:
            return {
                'success': False,
                'action': action_name,
                'reason': 'goal_rejected',
                'elapsed_s': time.monotonic() - started,
                'dock_status': self.dock_status(),
            }

        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future, deadline):
            try:
                cancel_future = goal_handle.cancel_goal_async()
                self._wait_future(cancel_future, time.monotonic() + 2.0)
            except Exception:  # noqa: BLE001 - best-effort cancellation
                pass
            return {
                'success': False,
                'action': action_name,
                'reason': 'action_timeout',
                'elapsed_s': time.monotonic() - started,
                'dock_status': self.dock_status(),
            }

        try:
            action_status = int(result_future.result().status)
        except Exception as exc:  # noqa: BLE001 - report ROS action failure to parent
            return {
                'success': False,
                'action': action_name,
                'reason': f'action_result_error:{exc!r}',
                'elapsed_s': time.monotonic() - started,
                'dock_status': self.dock_status(),
            }
        if action_status != GoalStatus.STATUS_SUCCEEDED:
            return {
                'success': False,
                'action': action_name,
                'reason': f'action_status:{action_status}',
                'action_status': action_status,
                'elapsed_s': time.monotonic() - started,
                'dock_status': self.dock_status(),
            }

        # The action result and /dock_status can arrive a little apart.  Require
        # both to agree so a following scenario never starts before physical dock.
        verify_deadline = min(deadline, time.monotonic() + 5.0)
        while time.monotonic() < verify_deadline:
            status = self.dock_status()
            if status is not None and status['is_docked'] == expected_docked:
                return {
                    'success': True,
                    'action': action_name,
                    'reason': None,
                    'action_status': action_status,
                    'elapsed_s': time.monotonic() - started,
                    'dock_status': status,
                }
            time.sleep(0.05)
        return {
            'success': False,
            'action': action_name,
            'reason': 'dock_status_not_confirmed',
            'action_status': action_status,
            'elapsed_s': time.monotonic() - started,
            'dock_status': self.dock_status(),
        }

    def dock(self, timeout=60.0):
        return self._run_dock_action('dock', self._dock_client, Dock.Goal(), True, timeout)

    def undock(self, timeout=30.0):
        return self._run_dock_action('undock', self._undock_client, Undock.Goal(), False, timeout)

    def ensure_docked(self, timeout=60.0, status_timeout=10.0):
        """Confirm the robot is on the charger, docking it first when necessary."""
        status = self.wait_for_dock_status(status_timeout)
        if status is None:
            return {
                'success': False,
                'action': 'ensure_docked',
                'reason': 'dock_status_unavailable',
                'dock_status': None,
            }
        if status['is_docked']:
            return {
                'success': True,
                'action': 'ensure_docked',
                'reason': None,
                'already_docked': True,
                'elapsed_s': 0.0,
                'dock_status': status,
            }
        return self.dock(timeout=timeout)

    def get_pose(self, t_os=None):
        """Return x, y and yaw in the odom frame (no SLAM required)."""
        try:
            target_time = Time(seconds=t_os) if t_os else Time()
            t = self.tf_buffer.lookup_transform(self.pose_frame, 'base_link',
                target_time, timeout=rclpy.duration.Duration(seconds=1.0))
            
            # 위치 추출
            x = t.transform.translation.x
            y = t.transform.translation.y
            
            # Quaternion -> Euler (Yaw) 변환
            q = t.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw_rad = math.atan2(siny_cosp, cosy_cosp)
            yaw_deg = math.degrees(yaw_rad)
            
            return round(x, 3), round(y, 3), round(yaw_deg, 2)
        except Exception as e:
            return None

    def control(self, linear_x=0.0, angular_z=0.0, duration=1.0):
        """선속도와 각속도를 조합하여 제어"""
        msg = TwistStamped()
        msg.header.frame_id = 'base_link'
        
        end_time = time.time() + duration
        while time.time() < end_time:
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x = float(linear_x)
            msg.twist.angular.z = float(angular_z) # 라디안/초
            self.pub.publish(msg)
            time.sleep(0.1)
        
        # 정지
        self.stop()

    def stop(self):
        msg = TwistStamped()
        msg.header.frame_id = 'base_link'
        for _ in range(3):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            time.sleep(0.05)

#%%
if __name__ =="__main__":
    rclpy.init()
    node = TB4AdvancedControl('robot_move_node')
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    print("Odometry 데이터 수신 대기...")
    while node.get_pose() is None: time.sleep(0.5)

    # 1. 현재 위치와 방향 확인
    start_pose = node.get_pose()
    print(f"시작 포즈: X={start_pose[0]}, Y={start_pose[1]}, Yaw={start_pose[2]}°")

    # 2. 이동 시나리오 테스트
    print("제자리 회전 테스트 (90도 방향으로)")
    node.control(angular_z=math.pi / 2, duration=1.0) # 초당 pi/2, 1초

    print("전진 테스트")
    node.control(linear_x=0.15, duration=2.0) # 초당 0.15, 2초

    # 3. 최종 결과
    end_pose = node.get_pose()
    print(f"최종 포즈: X={end_pose[0]}, Y={end_pose[1]}, Yaw={end_pose[2]}°")

    import random
    import math

    # 설정 파라미터
    SIGMA = 0.1  # 표준편차 (이 값이 클수록 원점에서 더 멀리까지 돌아다닙니다)
    SPEED = 0.15 # 이동 속도 (m/s)

    start_pose = node.get_pose()
    start_x, start_y, start_z = start_pose

    i = 0
    try:
        while True: # for i in range(15):
            # 1. 현재 포즈 확인
            start_pose = node.get_pose()
            if start_pose is None: continue
            curr_x, curr_y, curr_yaw = start_pose

            # 2. 다음 목적지를 가우시안 분포(평균 0, 표준편차 SIGMA)로 샘플링
            # 원점(0,0) 주변에서 포인트가 잡힐 확률이 가장 높음
            target_x = random.gauss(start_x, SIGMA)
            target_y = random.gauss(start_y, SIGMA)
            
            # 3. 목적지까지의 거리와 각도 계산
            dx = target_x - curr_x
            dy = target_y - curr_y
            distance = math.sqrt(dx**2 + dy**2)
            target_angle = math.degrees(math.atan2(dy, dx))
            
            # 4. 회전 제어: 목적지 방향과 현재 방향의 차이만큼 회전
            angle_diff = (target_angle - curr_yaw + 180) % 360 - 180
            turn_duration = abs(angle_diff) / 30.0  # 초당 30도 속도로 회전한다고 가정
            
            print(f"[{i+1}] 목표 지점 생성: ({target_x:.2f}, {target_y:.2f})")
            print(f"    ㄴ 회전 중: {angle_diff:.1f}도")
            node.control(linear_x=0.0, angular_z=(0.5 if angle_diff > 0 else -0.5), duration=turn_duration)
            
            # 5. 전진 제어: 목적지까지의 거리만큼 이동 (최대 1.5m 제한으로 안전 확보)
            move_dist = min(distance, 1.5)
            move_duration = move_dist / SPEED
            
            print(f"    ㄴ 전진 중: {move_dist:.2f}m")
            node.control(linear_x=SPEED, angular_z=0.0, duration=move_duration)
            
            time.sleep(1.0) # 위치 정착 및 데이터 수집 대기
            i += 1

    except KeyboardInterrupt:
        print("\n중단됨")

    rclpy.shutdown()

# %%
