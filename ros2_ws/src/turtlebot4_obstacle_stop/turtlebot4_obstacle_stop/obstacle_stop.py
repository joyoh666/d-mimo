"""Drive slowly until the TurtleBot 4 Lite sees an obstacle ahead."""

import math
import os

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ObstacleStop(Node):
    """Subscribe to LiDAR scans and publish a simple velocity command."""

    def __init__(self) -> None:
        # Every ROS 2 Python node inherits from Node and has a unique name.
        super().__init__("obstacle_stop")

        # Parameters let us tune behavior without editing the source code.
        self.declare_parameter("stop_distance", 0.50)
        self.declare_parameter("forward_speed", 0.10)
        self.declare_parameter("front_angle", 30.0)

        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.forward_speed = float(self.get_parameter("forward_speed").value)
        front_angle_degrees = float(self.get_parameter("front_angle").value)
        self.front_angle = math.radians(front_angle_degrees)

        # TurtleBot 4 uses Twist on Humble and TwistStamped on Jazzy.
        self.use_stamped = os.environ.get("ROS_DISTRO", "").lower() == "jazzy"
        command_type = TwistStamped if self.use_stamped else Twist

        self.cmd_vel_publisher = self.create_publisher(
            command_type,
            "/cmd_vel",
            10,
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Ready: stop_distance={self.stop_distance:.2f} m, "
            f"forward_speed={self.forward_speed:.2f} m/s"
        )

    def scan_callback(self, scan: LaserScan) -> None:
        """Run whenever a new LaserScan message arrives."""
        front_ranges = []

        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            is_in_front = abs(angle) <= self.front_angle

            if not is_in_front or math.isnan(distance):
                continue

            # Many LiDAR drivers use +inf to mean "nothing detected in range".
            if math.isinf(distance) and distance > 0.0:
                front_ranges.append(scan.range_max)
            elif scan.range_min <= distance <= scan.range_max:
                front_ranges.append(distance)

        # No valid measurement is treated as unsafe, so the robot stops.
        nearest = min(front_ranges) if front_ranges else 0.0

        if nearest > self.stop_distance:
            self.publish_velocity(self.forward_speed)
            self.get_logger().info(
                f"Clear ({nearest:.2f} m): moving forward",
                throttle_duration_sec=1.0,
            )
        else:
            self.publish_velocity(0.0)
            self.get_logger().warn(
                f"Obstacle ({nearest:.2f} m): stopped",
                throttle_duration_sec=1.0,
            )

    def publish_velocity(self, linear_x: float) -> None:
        """Create and publish the velocity message expected by this ROS version."""
        if self.use_stamped:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.twist.linear.x = linear_x
        else:
            message = Twist()
            message.linear.x = linear_x

        self.cmd_vel_publisher.publish(message)

    def stop(self) -> None:
        """Send a zero velocity before shutting down."""
        self.publish_velocity(0.0)


def main(args=None) -> None:
    """Initialize ROS, run the node, and stop safely on exit."""
    rclpy.init(args=args)
    node = ObstacleStop()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
