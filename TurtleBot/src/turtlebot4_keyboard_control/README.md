# TurtleBot 4 Lite arrow-key control

ROS 2 node that keeps an interactive keyboard-control session active until
`Q` or `Ctrl-C` is pressed.

## Controls

- Up / Down: move forward / backward
- Left / Right: rotate left / right
- Space: stop immediately
- `Q` or `Ctrl-C`: stop and end the experiment
- No movement key for 0.6 seconds: automatic safety stop

The defaults are deliberately conservative: `0.15 m/s` linear speed and
`0.60 rad/s` angular speed.

## Build

```bash
cd /home/osh666/d-mimo/TurtleBot
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

The final `source` command is required in every new terminal. Confirm the
Python import path before running a node:

```bash
python3 -c "import rclpy, turtlebot4_keyboard_control; print('imports OK')"
```

## Check robot connectivity

The PC and TurtleBot must use the same ROS distribution, DDS implementation,
and `ROS_DOMAIN_ID`. Before moving the robot, check the velocity topic:

```bash
ros2 topic info /cmd_vel --verbose
```

For ROS 2 Jazzy, `/cmd_vel` must accept
`geometry_msgs/msg/TwistStamped`.

## Run

Run the node in an interactive WSL, Ubuntu, or SSH terminal:

```bash
ros2 run turtlebot4_keyboard_control keyboard_control
```

After the workspace has been sourced, the equivalent Python package command is
also available:

```bash
python3 -m turtlebot4_keyboard_control
```

Use lower speeds for the first physical test:

```bash
ros2 run turtlebot4_keyboard_control keyboard_control --ros-args \
  -p linear_speed:=0.10 \
  -p angular_speed:=0.40 \
  -p key_timeout:=0.60
```

## Run independently alongside the USRP

Use two interactive terminals on the host PC. Keep the TurtleBot controller in
the foreground because it reads directly from that terminal.

Terminal 1 — TurtleBot:

```bash
cd /home/osh666/d-mimo/TurtleBot
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run turtlebot4_keyboard_control keyboard_control --ros-args \
  -p linear_speed:=0.10 \
  -p angular_speed:=0.40
```

Terminal 2 — USRP (using the existing USRP Python environment):

```bash
cd /home/osh666/d-mimo/USRP
python3 revised_ofdm_channel_capture_virtual7_timeavg.py
```

These processes run independently. This is suitable for collecting an initial
CSI dataset while driving, but their current timestamps must not be treated as
time-synchronized measurements.

## ROS 2 Humble compatibility

If the robot runs ROS 2 Humble rather than Jazzy, publish the older `Twist`
message type:

```bash
ros2 run turtlebot4_keyboard_control keyboard_control --ros-args \
  -p stamped:=false
```

Do not redirect standard input. Arrow keys are read directly from the active
terminal. Keep the robot in sight and keep a hand ready on Space.

## Lightring button example

The former standalone `TurtleBot/test.py` example is installed as a regular
ROS node, so its `irobot_create_msgs` dependency and Python import path are
handled by `rosdep` and `colcon`:

```bash
ros2 run turtlebot4_keyboard_control lightring_control
```
