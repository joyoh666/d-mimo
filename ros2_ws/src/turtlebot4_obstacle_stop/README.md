# TurtleBot 4 Lite obstacle-stop example

This beginner ROS 2 Python node demonstrates the basic data flow:

```text
/scan (LaserScan) -> scan_callback() -> decision -> /cmd_vel
```

The robot moves forward at `0.10 m/s` while the nearest valid LiDAR point in
the front 60-degree area is farther than `0.50 m`. It stops otherwise.

## What to look for in the code

Open `turtlebot4_obstacle_stop/obstacle_stop.py` and follow these parts:

1. `ObstacleStop(Node)`: defines a ROS 2 node.
2. `create_subscription(...)`: receives `/scan` LiDAR messages.
3. `scan_callback(...)`: turns sensor data into a decision.
4. `create_publisher(...)`: prepares the `/cmd_vel` output.
5. `publish_velocity(...)`: sends the motion command.
6. `rclpy.spin(...)`: keeps the node alive and processes callbacks.

## Copy from the Mac to the robot

From the repository root on the Mac, replace `turtlebot4` with the SSH host or
IP address you use for the robot:

```bash
rsync -av ros2_ws/src/turtlebot4_obstacle_stop/ \
  turtlebot4:/home/ubuntu/turtlebot4_ws/src/turtlebot4_obstacle_stop/
```

## Build on the robot

```bash
ssh turtlebot4
cd ~/turtlebot4_ws
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select turtlebot4_obstacle_stop
source install/setup.bash
```

## Run safely

Put the robot on a clear, flat floor and stay ready to stop it. In the robot's
SSH terminal:

```bash
ros2 run turtlebot4_obstacle_stop obstacle_stop
```

Press `Ctrl-C` to stop. To change the threshold and speed without editing code:

```bash
ros2 run turtlebot4_obstacle_stop obstacle_stop --ros-args \
  -p stop_distance:=0.70 \
  -p forward_speed:=0.05
```

Before allowing motion, you can confirm the topics and message type:

```bash
ros2 topic echo /scan --once
ros2 topic info /cmd_vel --verbose
```
