from glob import glob

from setuptools import find_packages, setup

package_name = "roboto_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/models/roboto_bot",
         glob("models/roboto_bot/*.sdf") + glob("models/roboto_bot/*.config")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hem",
    maintainer_email="therbligsproject2@gmail.com",
    description="ROS 2 layer over roboto_core.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "roboto_node = roboto_ros.roboto_node:main",
        ],
    },
)
