"""ROS 2 node for safely controlling a TurtleBot 4 with arrow keys."""

from __future__ import annotations

import time

from geometry_msgs.msg import Twist, TwistStamped
import rclpy
from rclpy.node import Node

from .key_input import TerminalKeyboard
from .motion import Motion, motion_for_key, STOP


HELP_TEXT = """
TurtleBot 4 Lite keyboard control
---------------------------------
  Up arrow     : move forward
  Down arrow   : move backward
  Left arrow   : rotate left
  Right arrow  : rotate right
  Space        : stop immediately
  Q / Ctrl-C   : stop and finish the experiment

Hold or repeatedly press an arrow key to keep moving. If no movement key is
received before the safety timeout, the robot stops automatically.
"""


class KeyboardControl(Node):
    """Publish velocity commands from non-blocking terminal input."""

    def __init__(self, keyboard: TerminalKeyboard) -> None:
        super().__init__('turtlebot4_keyboard_control')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('angular_speed', 0.60)
        self.declare_parameter('key_timeout', 0.60)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('stamped', True)
        self.declare_parameter('frame_id', 'base_link')

        self._topic = (
            self.get_parameter('cmd_vel_topic')
            .get_parameter_value()
            .string_value
        )
        self._linear_speed = (
            self.get_parameter('linear_speed')
            .get_parameter_value()
            .double_value
        )
        self._angular_speed = (
            self.get_parameter('angular_speed')
            .get_parameter_value()
            .double_value
        )
        self._key_timeout = (
            self.get_parameter('key_timeout')
            .get_parameter_value()
            .double_value
        )
        publish_rate = (
            self.get_parameter('publish_rate')
            .get_parameter_value()
            .double_value
        )
        self._stamped = (
            self.get_parameter('stamped')
            .get_parameter_value()
            .bool_value
        )
        self._frame_id = (
            self.get_parameter('frame_id')
            .get_parameter_value()
            .string_value
        )

        if self._linear_speed <= 0.0:
            raise ValueError('linear_speed must be greater than zero')
        if self._angular_speed <= 0.0:
            raise ValueError('angular_speed must be greater than zero')
        if self._key_timeout <= 0.0:
            raise ValueError('key_timeout must be greater than zero')
        if publish_rate <= 0.0:
            raise ValueError('publish_rate must be greater than zero')

        message_type = TwistStamped if self._stamped else Twist
        self._publisher = self.create_publisher(
            message_type,
            self._topic,
            10,
        )
        self._keyboard = keyboard
        self._motion = STOP
        self._last_motion_key_ns: int | None = None
        self._timeout_stop_sent = False
        self.quit_requested = False
        self._timer = self.create_timer(1.0 / publish_rate, self._update)

        selected_type = 'TwistStamped' if self._stamped else 'Twist'
        self.get_logger().info(
            f'Publishing {selected_type} on {self._topic}; '
            f'linear={self._linear_speed:.2f} m/s, '
            f'angular={self._angular_speed:.2f} rad/s, '
            f'timeout={self._key_timeout:.2f} s'
        )

    def _update(self) -> None:
        """Process keys, enforce the watchdog, and publish motion."""
        now_ns = time.monotonic_ns()

        for key in self._keyboard.read_available():
            if key == 'quit':
                self._motion = STOP
                self.publish_stop()
                self.quit_requested = True
                return

            self._motion = motion_for_key(
                key,
                linear_speed=self._linear_speed,
                angular_speed=self._angular_speed,
            )

            if self._motion.is_stopped:
                self._last_motion_key_ns = None
                self._timeout_stop_sent = True
                self._publish(STOP)
            else:
                self._last_motion_key_ns = now_ns
                self._timeout_stop_sent = False
                self._publish(self._motion)

        if self._motion.is_stopped or self._last_motion_key_ns is None:
            return

        elapsed = (now_ns - self._last_motion_key_ns) / 1_000_000_000.0
        if elapsed >= self._key_timeout:
            self._motion = STOP
            if not self._timeout_stop_sent:
                self._publish(STOP)
                self._timeout_stop_sent = True
            return

        # Repeat the latest command to keep the base velocity watchdog fed.
        self._publish(self._motion)

    def _publish(self, motion: Motion) -> None:
        """Convert planar motion to the configured ROS message type."""
        if self._stamped:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self._frame_id
            twist = message.twist
        else:
            message = Twist()
            twist = message

        twist.linear.x = motion.linear_x
        twist.angular.z = motion.angular_z
        self._publisher.publish(message)

    def publish_stop(self) -> None:
        """Publish redundant zero commands during shutdown."""
        for _ in range(3):
            self._publish(STOP)


def main(args: list[str] | None = None) -> None:
    """Run keyboard control until Q, Ctrl-C, or ROS shutdown."""
    rclpy.init(args=args)
    node: KeyboardControl | None = None

    try:
        with TerminalKeyboard() as keyboard:
            node = KeyboardControl(keyboard)
            print(HELP_TEXT, flush=True)
            while rclpy.ok() and not node.quit_requested:
                rclpy.spin_once(node, timeout_sec=0.10)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.publish_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
