"""Key-to-motion mapping kept separate for easy testing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Motion:
    """Planar velocity command in SI units."""

    linear_x: float = 0.0
    angular_z: float = 0.0

    @property
    def is_stopped(self) -> bool:
        return self.linear_x == 0.0 and self.angular_z == 0.0


STOP = Motion()


def motion_for_key(key: str, linear_speed: float, angular_speed: float) -> Motion:
    """Map one decoded key to a TurtleBot planar velocity."""
    motions = {
        "up": Motion(linear_x=linear_speed),
        "down": Motion(linear_x=-linear_speed),
        "left": Motion(angular_z=angular_speed),
        "right": Motion(angular_z=-angular_speed),
        "stop": STOP,
    }
    return motions.get(key, STOP)
