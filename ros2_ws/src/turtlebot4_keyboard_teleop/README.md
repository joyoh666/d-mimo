# TurtleBot 4 Lite arrow-key teleoperation

ROS 2 Python node that drives a TurtleBot 4 Lite through `/cmd_vel`.

- Arrow Up/Down: forward/reverse
- Arrow Left/Right: rotate left/right
- Space: stop immediately
- `Q` or `Ctrl-C`: stop and exit
- No motion key for 0.6 seconds: automatic stop

The default speeds are deliberately conservative: `0.20 m/s` linear and
`0.80 rad/s` angular. Keep the robot in sight and test with the wheels clear
of obstacles first.

## Supported ROS 2 versions

The node supports both current TurtleBot 4 software combinations:

- ROS 2 Humble: publishes `geometry_msgs/msg/Twist`
- ROS 2 Jazzy: publishes `geometry_msgs/msg/TwistStamped`

With `message_type:=auto` (the default), `ROS_DISTRO` selects the correct
message. It can be overridden with `twist` or `twist_stamped`.

## Build

Run these commands on the TurtleBot or on an Ubuntu PC configured to
communicate with it. Replace `humble` with `jazzy` when applicable.

```bash
cd /path/to/d-mimo/ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The controlling PC and TurtleBot must use matching ROS 2 distributions,
`ROS_DOMAIN_ID`, and DDS configuration. Confirm connectivity before driving:

```bash
ros2 topic list
ros2 topic info /cmd_vel --verbose
```

## Run

Run directly in an interactive terminal (recommended):

```bash
ros2 run turtlebot4_keyboard_teleop keyboard_teleop
```

Custom speed example:

```bash
ros2 run turtlebot4_keyboard_teleop keyboard_teleop --ros-args \
  -p linear_speed:=0.15 \
  -p angular_speed:=0.60 \
  -p command_timeout:=0.60
```

If automatic message selection does not match a customized robot stack:

```bash
# Humble-style command
ros2 run turtlebot4_keyboard_teleop keyboard_teleop --ros-args \
  -p message_type:=twist

# Jazzy-style command
ros2 run turtlebot4_keyboard_teleop keyboard_teleop --ros-args \
  -p message_type:=twist_stamped
```

Do not redirect standard input or run the executable without a terminal;
arrow keys are read directly from the active terminal.
