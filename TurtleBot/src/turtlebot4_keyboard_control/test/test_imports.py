"""Import regression tests for the TurtleBot ROS package."""

import importlib
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch


def fake_ros_modules() -> dict[str, types.ModuleType]:
    """Return minimal ROS modules for import-only tests."""

    class Message:
        pass

    class Node:
        pass

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Twist = Message
    geometry_msgs_msg.TwistStamped = Message
    geometry_msgs.msg = geometry_msgs_msg

    irobot_create_msgs = types.ModuleType("irobot_create_msgs")
    irobot_create_msgs_msg = types.ModuleType("irobot_create_msgs.msg")
    irobot_create_msgs_msg.InterfaceButtons = Message
    irobot_create_msgs_msg.LightringLeds = Message
    irobot_create_msgs.msg = irobot_create_msgs_msg

    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = Node
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.qos_profile_sensor_data = object()
    rclpy.node = rclpy_node
    rclpy.qos = rclpy_qos

    return {
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "irobot_create_msgs": irobot_create_msgs,
        "irobot_create_msgs.msg": irobot_create_msgs_msg,
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
    }


class PackageImportTest(unittest.TestCase):
    def test_pure_python_modules_import(self) -> None:
        importlib.import_module("turtlebot4_keyboard_control.key_input")
        importlib.import_module("turtlebot4_keyboard_control.motion")

    def test_ros_nodes_use_package_imports(self) -> None:
        module_names = (
            "turtlebot4_keyboard_control.keyboard_control",
            "turtlebot4_keyboard_control.lightring_control",
        )
        for module_name in module_names:
            sys.modules.pop(module_name, None)

        with patch.dict(sys.modules, fake_ros_modules()):
            for module_name in module_names:
                module = importlib.import_module(module_name)
                self.assertTrue(callable(module.main))

        for module_name in module_names:
            sys.modules.pop(module_name, None)

    def test_keyboard_node_supports_direct_script_imports(self) -> None:
        package_dir = (
            Path(__file__).resolve().parents[1]
            / "turtlebot4_keyboard_control"
        )
        with (
            patch.dict(sys.modules, fake_ros_modules()),
            patch.object(sys, "path", [str(package_dir), *sys.path]),
        ):
            namespace = runpy.run_path(
                str(package_dir / "keyboard_control.py"),
                run_name="direct_import_test",
            )
        self.assertTrue(callable(namespace["main"]))
        sys.modules.pop("key_input", None)
        sys.modules.pop("motion", None)


if __name__ == "__main__":
    unittest.main()
