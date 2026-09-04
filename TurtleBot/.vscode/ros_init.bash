# Load the normal interactive shell configuration first.
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# ROS 2 Jazzy matches Ubuntu 24.04 and the current TurtleBot 4 image.
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
