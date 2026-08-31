"""ROS 2 node for driving TurtleBot 4 Lite with the arrow keys."""

from __future__ import annotations

import os
import time

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node

from .key_input import TerminalKeyboard
from .motion import Motion, STOP, motion_for_key


HELP_TEXT = """
TurtleBot 4 Lite keyboard control
---------------------------------
  Up arrow     : forward
  Down arrow   : reverse
  Left arrow   : rotate left
  Right arrow  : rotate right
  Space        : stop immediately
  Q / Ctrl-C   : stop and quit

Keep tapping or hold an arrow key. Releasing it triggers the safety timeout.
Keep the robot in sight and be ready to press Space.
"""


class KeyboardTeleop(Node):
    """Publish safe velocity commands based on non-blocking terminal input."""

    def __init__(self, keyboard: TerminalKeyboard) -> None:
        super().__init__("turtlebot4_keyboard_teleop")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("linear_speed", 0.20)
        self.declare_parameter("angular_speed", 0.80)
        self.declare_parameter("command_timeout", 0.60)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("message_type", "auto")

        self._topic = str(self.get_parameter("cmd_vel_topic").value)
        self._linear_speed = float(self.get_parameter("linear_speed").value)
        self._angular_speed = float(self.get_parameter("angular_speed").value)
        self._command_timeout = float(
            self.get_parameter("command_timeout").value
        )
        publish_rate = float(self.get_parameter("publish_rate").value)
        configured_type = str(self.get_parameter("message_type").value).lower()

        if self._linear_speed <= 0.0:
            raise ValueError("linear_speed must be greater than zero")
        if self._angular_speed <= 0.0:
            raise ValueError("angular_speed must be greater than zero")
        if self._command_timeout <= 0.0:
            raise ValueError("command_timeout must be greater than zero")
        if publish_rate <= 0.0:
            raise ValueError("publish_rate must be greater than zero")

        self._use_stamped = self._resolve_message_type(configured_type)
        message_class = TwistStamped if self._use_stamped else Twist
        self._publisher = self.create_publisher(message_class, self._topic, 10)

        self._keyboard = keyboard
        self._motion = STOP
        self._last_motion_key_ns: int | None = None
        self._sent_timeout_stop = False
        self.quit_requested = False
        self._timer = self.create_timer(1.0 / publish_rate, self._update)

        selected_type = "TwistStamped" if self._use_stamped else "Twist"
        self.get_logger().info(
            f"Publishing {selected_type} on {self._topic} "
            f"(linear={self._linear_speed:.2f} m/s, "
            f"angular={self._angular_speed:.2f} rad/s)"
        )

    @staticmethod
    def _resolve_message_type(configured_type: str) -> bool:
        valid_types = {"auto", "twist", "twist_stamped"}
        if configured_type not in valid_types:
            raise ValueError(
                "message_type must be one of: auto, twist, twist_stamped"
            )
        if configured_type == "twist_stamped":
            return True
        if configured_type == "twist":
            return False

        ros_distro = os.environ.get("ROS_DISTRO", "").lower()
        # TurtleBot 4 uses stamped velocity commands on Jazzy and newer releases.
        return ros_distro not in {"galactic", "humble"}

    def _update(self) -> None:
        # A monotonic wall clock keeps the safety timeout working even when a
        # simulation clock is paused or reset.
        now_ns = time.monotonic_ns()
        for key in self._keyboard.read_available():
            if key == "quit":
                self._motion = STOP
                self.publish_stop()
                self.quit_requested = True
                return

            self._motion = motion_for_key(
                key,
                linear_speed=self._linear_speed,
                angular_speed=self._angular_speed,
            )
            self._last_motion_key_ns = now_ns
            self._sent_timeout_stop = self._motion.is_stopped
            self._publish(self._motion)

        if self._motion.is_stopped or self._last_motion_key_ns is None:
            return

        elapsed = (now_ns - self._last_motion_key_ns) / 1_000_000_000.0
        if elapsed >= self._command_timeout:
            self._motion = STOP
            if not self._sent_timeout_stop:
                self._publish(STOP)
                self._sent_timeout_stop = True
            return

        # Repeat the latest command so the Create 3 velocity watchdog stays fed.
        self._publish(self._motion)

    def _publish(self, motion: Motion) -> None:
        if self._use_stamped:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            twist = message.twist
        else:
            message = Twist()
            twist = message

        twist.linear.x = motion.linear_x
        twist.angular.z = motion.angular_z
        self._publisher.publish(message)

    def publish_stop(self) -> None:
        """Publish redundant zero commands for a safer shutdown."""
        for _ in range(3):
            self._publish(STOP)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: KeyboardTeleop | None = None

    try:
        with TerminalKeyboard() as keyboard:
            node = KeyboardTeleop(keyboard)
            print(HELP_TEXT, flush=True)
            while rclpy.ok() and not node.quit_requested:
                rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.publish_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
