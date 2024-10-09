import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # # Get the path to the config file
    # config_file = os.path.join(
    #     get_package_share_directory('monogs_ros'),
    #     'configs',
    #     'config.yaml'
    # )

    # Define the node and load the config file
    return LaunchDescription([
        Node(
            package='monogs_ros',
            executable='slam_frontend',
            name='frontend_node',
            output='screen',
            # parameters=[config_file]  # Load the YAML config file
        ),
    ])