from setuptools import find_packages, setup

package_name = "roboto_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "scipy", "pyyaml"],
    zip_safe=True,
    maintainer="hem",
    maintainer_email="therbligsproject2@gmail.com",
    description=(
        "Pure-Python core for GIS-prior SLAM: frames, occupancy mapping, scan "
        "matching, planning, and map discrepancy detection. No ROS imports -- "
        "so it is unit-testable without a ROS environment or a simulator."
    ),
    license="MIT",
    tests_require=["pytest"],
)
