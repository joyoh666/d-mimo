# TurtleBot ROS 2 workspace

This directory is a ROS 2 workspace. The Python package lives below
`src/turtlebot4_keyboard_control`, so it is not importable from the repository
root until the workspace has been built and its environment has been sourced.

## Build and activate the package

```bash
cd TurtleBot
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Verify that both ROS and the local package are on the active Python path:

```bash
python3 -c "import rclpy, turtlebot4_keyboard_control; print('imports OK')"
```

The `source install/setup.bash` command applies only to the current terminal.
Run it again in every new terminal before importing the package or using
`ros2 run`.

## Development tests before a ROS build

The workspace-level `pytest.ini` adds the package source directory only while
tests run; the application itself still uses the normal ROS installation.

```bash
cd TurtleBot
python3 -m pytest
```

For VS Code/Pylance, open this `TurtleBot` directory as the workspace and start
the editor from a terminal where `/opt/ros/jazzy/setup.bash` has been sourced.
`pyrightconfig.json` supplies the local package path; sourcing ROS supplies
`rclpy`, `geometry_msgs`, and `irobot_create_msgs`.

See `src/turtlebot4_keyboard_control/README.md` for controls and run commands.
