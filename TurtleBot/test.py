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


if __name__ == "__main__":
    main()
