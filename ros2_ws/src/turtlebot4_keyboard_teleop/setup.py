from setuptools import find_packages, setup


package_name = "turtlebot4_keyboard_teleop"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TurtleBot 4 User",
    maintainer_email="user@example.com",
    description="Safe arrow-key teleoperation for TurtleBot 4 Lite.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "keyboard_teleop = turtlebot4_keyboard_teleop.keyboard_teleop:main",
        ],
    },
)
