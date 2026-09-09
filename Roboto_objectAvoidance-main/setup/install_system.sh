#!/usr/bin/env bash
# Install ROS 2 Humble + Gazebo Fortress on Ubuntu 22.04 (WSL2).
#
# Run as root inside the distro:
#     wsl -d Ubuntu-22.04 -u root -- bash /mnt/e/roboto/setup/install_system.sh
#
# Idempotent: safe to re-run. Reproducibility artifact for the report --
# this file is the authoritative record of the system environment.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
CODENAME=jammy          # Ubuntu 22.04, the only release Humble targets

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

log "Base tooling"
apt-get update -qq
apt-get install -y -qq \
    curl gnupg lsb-release software-properties-common locales ca-certificates

log "UTF-8 locale (ROS 2 requires it)"
locale-gen en_US en_US.UTF-8 >/dev/null
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

log "Enabling the universe repository"
add-apt-repository -y universe >/dev/null

log "ROS 2 apt repository"
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu ${CODENAME} main" \
    > /etc/apt/sources.list.d/ros2.list

log "OSRF Gazebo apt repository"
curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable ${CODENAME} main" \
    > /etc/apt/sources.list.d/gazebo-stable.list

apt-get update -qq

log "ROS 2 Humble desktop (this is the long one, ~2 GB)"
apt-get install -y -qq ros-humble-desktop ros-dev-tools

log "Gazebo Fortress + the ROS bridge"
# ros-humble-ros-gz is the Humble<->Fortress bridge; ignition-fortress brings
# the `ign gazebo` CLI and the demo worlds used by the Gate A rendering test.
apt-get install -y -qq ignition-fortress ros-humble-ros-gz

log "Navigation + SLAM baseline"
# slam_toolbox is installed as an EVALUATION BASELINE to compare our
# from-scratch SLAM against -- not as the project's SLAM implementation.
apt-get install -y -qq \
    ros-humble-nav2-controller \
    ros-humble-nav2-regulated-pure-pursuit-controller \
    ros-humble-nav2-costmap-2d \
    ros-humble-nav2-msgs \
    ros-humble-nav2-util \
    ros-humble-slam-toolbox \
    ros-humble-tf-transformations \
    ros-humble-teleop-twist-keyboard \
    ros-humble-rmw-cyclonedds-cpp

log "Python + build tooling"
apt-get install -y -qq \
    python3-colcon-common-extensions python3-rosdep python3-vcstool \
    python3-pip python3-pytest python3-numpy python3-scipy python3-yaml \
    git mesa-utils

log "rosdep"
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    rosdep init >/dev/null
fi

log "Versions"
{
    echo "date          : $(date -Iseconds)"
    echo "ubuntu        : $(lsb_release -ds)"
    echo "kernel        : $(uname -r)"
    echo "ros_distro    : humble"
    echo "ros_desktop   : $(dpkg -query -W -f='${Version}' ros-humble-desktop 2>/dev/null || echo MISSING)"
    echo "ignition      : $(dpkg-query -W -f='${Version}' ignition-fortress 2>/dev/null || echo MISSING)"
    echo "ros_gz        : $(dpkg-query -W -f='${Version}' ros-humble-ros-gz 2>/dev/null || echo MISSING)"
    echo "slam_toolbox  : $(dpkg-query -W -f='${Version}' ros-humble-slam-toolbox 2>/dev/null || echo MISSING)"
    echo "gl_renderer   : $(glxinfo -B 2>/dev/null | sed -n 's/^OpenGL renderer string: //p')"
    echo "gl_version    : $(glxinfo -B 2>/dev/null | sed -n 's/^OpenGL core profile version string: //p')"
} | tee /etc/roboto-env.txt

log "Done. 'source /opt/ros/humble/setup.bash' to use ROS 2."
