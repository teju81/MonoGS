#!/bin/bash

colcon build
source install/setup.bash
ros2 run monogs_ros slam_backend