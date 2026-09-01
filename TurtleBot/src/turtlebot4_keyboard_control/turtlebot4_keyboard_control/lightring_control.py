"""ROS 2 node that changes the TurtleBot 4 lightring from button input."""

from __future__ import annotations

from irobot_create_msgs.msg import InterfaceButtons, LightringLeds
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class LightringControl(Node):
    """Publish a six-color lightring pattern when button 1 is pressed."""

    def __init__(self) -> None:
        super().__init__("turtlebot4_lightring_control")
        self._lights_on = False

        self._button_subscription = self.create_subscription(
            InterfaceButtons,
            "/interface_buttons",
            self._interface_buttons_callback,
            qos_profile_sensor_data,
        )
        self._lightring_publisher = self.create_publisher(
            LightringLeds,
            "/cmd_lightring",
            qos_profile_sensor_data,
        )

    def _interface_buttons_callback(self, message: InterfaceButtons) -> None:
        if message.button_1.is_pressed and not self._lights_on:
            self.get_logger().info("Button 1 is pressed")
            self._publish_color_pattern()

    def _publish_color_pattern(self) -> None:
        message = LightringLeds()
        message.header.stamp = self.get_clock().now().to_msg()
        message.override_system = True

        colors = (
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        )
        for led, (red, green, blue) in zip(message.leds, colors):
            led.red = red
            led.green = green
            led.blue = blue

        self._lightring_publisher.publish(message)
        self._lights_on = True


def main(args: list[str] | None = None) -> None:
    """Run the lightring node until ROS shutdown."""
    rclpy.init(args=args)
    node = LightringControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
