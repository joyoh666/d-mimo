from setuptools import find_packages, setup


package_name = 'turtlebot4_keyboard_control'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml', 'README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TurtleBot 4 User',
    maintainer_email='user@example.com',
    description='Safe arrow-key control for a TurtleBot 4 Lite.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'keyboard_control = '
            'turtlebot4_keyboard_control.keyboard_control:main',
            'lightring_control = '
            'turtlebot4_keyboard_control.lightring_control:main',
        ],
    },
)
