#!/bin/bash

# Name of the tmux session
SESSION="monogs_ros_session"

# ROS executables (replace these with your actual ROS executables)
RUN_FRONTEND="ros2 run monogs_ros slam_frontend"
RUN_BACKEND="ros2 run monogs_ros slam_backend"
RUN_GUI="ros2 run monogs_ros slam_gui"
RUN_LOOPCLOSING="ros2 run monogs_ros slam_loopclosing"

# Check if the tmux session already exists
tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
    # Step 1: Create a new tmux session and window
    tmux new-session -d -s $SESSION -n "ROS"

    # Step 2: Split the window into 4 panes (2x2 grid)
    tmux split-window -h  # Split horizontally
    tmux split-window -v  # Split the left pane vertically
    tmux select-pane -t 0  # Move focus to the first pane (top-left)
    tmux split-window -v  # Split the right pane vertically

    # Step 3: Send ROS commands to each pane
    tmux send-keys -t 0 "$RUN_FRONTEND" C-m  # Top-left pane
    tmux send-keys -t 1 "$RUN_BACKEND" C-m  # Top-right pane
    tmux send-keys -t 2 "$RUN_GUI" C-m  # Bottom-left pane
    tmux send-keys -t 3 "$RUN_LOOPCLOSING" C-m  # Bottom-right pane

    # Step 4: Attach to the tmux session - Cannot be done automatically from within a ros launch command (need to manually do it)
    echo "tmux session '$SESSION' created. Nodes are running in the background. You can manually attach using: tmux attach-session -t $SESSION if you want to see the output of the nodes on the terminal screen"
    # tmux attach-session -t $SESSION
else
    echo "Session $SESSION already exists.. Nodes are running in the background. You can manually attach using: tmux attach-session -t $SESSION if you want to see the output of the nodes on the terminal screen"
    # tmux attach-session -t $SESSION
fi