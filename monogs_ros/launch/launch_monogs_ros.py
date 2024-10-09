from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    # Define the path to the bash script
    bash_script_path = 'tmux_launch_all_nodes.sh'

    return LaunchDescription([
        # Execute the bash script that sets up tmux and runs ROS nodes with the config file
        ExecuteProcess(
            cmd=['bash', bash_script_path],
            output='screen'
        ),
    ])