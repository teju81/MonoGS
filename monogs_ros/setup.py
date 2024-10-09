from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'monogs_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        ('bin', glob('scripts/*.sh')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'monogs_launch = monogs_ros.slam:main',
            'slam_frontend = monogs_ros.utils.slam_frontend:main',
            'slam_backend = monogs_ros.utils.slam_backend:main',
            'slam_gui = monogs_ros.gui.slam_gui:main',
            'slam_loopclosing = monogs_ros.utils.slam_loopclosing:main',
        ],
    },
)
