import unittest

from turtlebot4_keyboard_teleop.motion import STOP, motion_for_key


class MotionTest(unittest.TestCase):
    def test_arrow_key_motion_mapping(self) -> None:
        self.assertEqual(motion_for_key("up", 0.2, 0.8).linear_x, 0.2)
        self.assertEqual(motion_for_key("down", 0.2, 0.8).linear_x, -0.2)
        self.assertEqual(motion_for_key("left", 0.2, 0.8).angular_z, 0.8)
        self.assertEqual(motion_for_key("right", 0.2, 0.8).angular_z, -0.8)

    def test_stop_and_unknown_key_are_stationary(self) -> None:
        self.assertEqual(motion_for_key("stop", 0.2, 0.8), STOP)
        self.assertEqual(motion_for_key("unknown", 0.2, 0.8), STOP)


if __name__ == "__main__":
    unittest.main()
