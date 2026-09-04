"""Compatibility entry point for the packaged lightring control node.

Prefer ``ros2 run turtlebot4_keyboard_control lightring_control`` after
building and sourcing the TurtleBot workspace.
"""

try:
    from turtlebot4_keyboard_control.lightring_control import main
except ModuleNotFoundError as error:
    missing_module = error.name or "a required ROS module"
    raise SystemExit(
        f"Cannot import {missing_module!r}. Build the TurtleBot workspace and "
        "source both /opt/ros/<distro>/setup.bash and install/setup.bash."
    ) from error

<<<<<<< HEAD:TurtleBot/test.py

if __name__ == "__main__":
    main()
=======
    def __init__(self):
        super().__init__('turtlebot4_first_python_node')

        self.interface_buttons_subscriber = self.create_subscription(
            InterfaceButtons,
            '/interface_buttons',
            self.interface_buttons_callback,
            qos_profile_sensor_data
        )

        self.lightring_publisher = self.create_publisher(
            LightringLeds,
            '/cmd_lightring',
            qos_profile_sensor_data
        )

    def interface_buttons_callback(self, create3_buttons_msg: InterfaceButtons):
        if create3_buttons_msg.button_1.is_pressed:
            self.get_logger().info("Button 1 is pressed!")    
            self.button_1_function()

    def button_1_function(self):
        # Create a ROS 2 message
        lightring_msg = LightringLeds()
        # Stamp the message with the current time
        lightring_msg.header.stamp = self.get_clock().now().to_msg()
        if not self.lights_on : 
            # Override system lights
            lightring_msg.override_system = True

            # LED 0
            lightring_msg.leds[0].red = 255
            lightring_msg.leds[0].blue = 0
            lightring_msg.leds[0].green = 0

            # LED 1
            lightring_msg.leds[1].red = 0
            lightring_msg.leds[1].blue = 255
            lightring_msg.leds[1].green = 0

            # LED 2
            lightring_msg.leds[2].red = 0
            lightring_msg.leds[2].blue = 0
            lightring_msg.leds[2].green = 255

            # LED 3
            lightring_msg.leds[3].red = 255
            lightring_msg.leds[3].blue = 255
            lightring_msg.leds[3].green = 0

            # LED 4
            lightring_msg.leds[4].red = 255
            lightring_msg.leds[4].blue = 0
            lightring_msg.leds[4].green = 255

            # LED 5
            lightring_msg.leds[5].red = 0
            lightring_msg.leds[5].blue = 255
            lightring_msg.leds[5].green = 255

            # Publish the message
            self.lightring_publisher.publish(lightring_msg)
            # Toggle the lights on status
            self.lights_on_ = not self.lights_on_
            
def main(args=None):
    rclpy.init(args=args)
    node = TurtleBot4FirstNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
>>>>>>> a416ca0 (0904):TurtleBot/Tutorial.py
