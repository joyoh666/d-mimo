'''
ros2 launch turtlebot4_bringup lite.launch.py
ros2 launch turtlebot4_navigation slam.launch.py

이 코드 실행 전 turtlebot Raspberry Pi 접속하셔서 위 코드 실행하세요
'''

#%%
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TwistStamped
from tf2_ros import TransformListener, Buffer
import time, threading
import math

class TB4AdvancedControl(Node):
    def __init__(self, name='tb4_advanced_control'):
        super().__init__(name)
        self.pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_pose(self, t_os=None):
        """특정 시점의 x, y, yaw(degree)를 반환"""
        try:
            target_time = Time(seconds=t_os) if t_os else Time()
            t = self.tf_buffer.lookup_transform('map', 'base_link', 
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

    print("SLAM 데이터 수신 대기...")
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
