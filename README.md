# SoTeleIK

**SoTeleIK** is a virtual simulation environment made using **ROS 2** and **MuJoCo**. The environment presents three **so_100** robotic arms controlled via inverse kinematics to follow teleoperation commands. The simulation applies the Levenberg-Marquardt algorithm for inverse kinematics and uses a PD controller for joint actuation. Workspace calibration allows mapping human teleoperator movements to the robot's reachable space.

---

## Features

* Complete integration with ROS 2 (Humble) middleware
* Integrates with the MuJoCo physics simulation environment
* Simulates three 6-DoF `so_100` robotic arms
* Inverse Kinematics using the Levenberg-Marquardt algorithm
* PD Control for joint torque actuation
* Workspace calibration for mapping teleoperation input (Human-to-Robot)
* Real-time camera view streaming via OpenCV

---

## Future Implementation

* Impementation of NGDF ([ Neural Grasp Distance Fields](https://sites.google.com/view/neural-grasp-distance-fields) ).
* Implementation of fullbody coupled IK for joint manipulator grasping.

---

## Requirements

* **ROS 2 Humble** (or potentially Foxy/Galactic)
* **MuJoCo Physics Engine** (and Python bindings: `mujoco`)
* **Python 3** with libraries:
    * `numpy`
    * `PyYAML`
    * `opencv-python` (`cv2`)
* **`teleop_interfaces` ROS 2 package** (for the `/meta/teleop_data_raw` message type)
* **`mujoco_menagerie`** downloaded (specifically the `trs_so_arm100` models)
* A ROS 2 package capable of publishing `teleop_interfaces/msg/TeleopDataRaw` messages.

---

## Directory Structure

### Src
```
── so100mjk
│   ├── bag_files
│   │   ├── headmoving.db3
│   │   ├── headstable.db3
│   │   ├── headtracking.db3
│   │   └── metadata.yaml
│   ├── config
│   │   ├── arm.yaml
│   │   └── calibrate.yaml
│   ├── launch
│   ├── package.xml
│   ├── resource
│   │   └── so100mjk
│   ├── setup.cfg
│   ├── setup.py
│   ├── so100mjk
│   │   ├── calibrate.py
│   │   ├── __init__.py
│   │   ├── lm_ik_parallel.py
│   │   ├── mujoco_node.py
│   │   └── __pycache__
│   │       ├── calibrate.cpython-310.pyc
│   │       ├── __init__.cpython-310.pyc
│   │       ├── lm_ik_parallel.cpython-310.pyc
│   │       └── mujoco_node.cpython-310.pyc
│   └── test
│       ├── test_copyright.py
│       ├── test_flake8.py
│       └── test_pep257.py
└── teleop_interfaces
    ├── action
    │   └── IK.action
    ├── CMakeLists.txt
    ├── msg
    │   ├── CartesianGroup.msg
    │   ├── JointGroup.msg
    │   ├── JoyPad.msg
    │   └── TeleopDataRaw.msg
    ├── package.xml
    └── teleop_interfaces
        └── __init__.py
```

### assets_calib
```
assets_calib/
├── calibrate_workspace.py
├── scene_triple.xml
├── so_arm100_2.xml
├── so_arm100_3.xml
└── so_arm100.xml
```

## Installation

```bash
# Create and initialize a ROS 2 workspace (if you don't have one)
mkdir -p ~/so_teleik_ws/src
cd ~/so_teleik_ws/src

git clone https://github.com/AkshayArjun/SoTeleIK.git


# Install Python dependencies
pip install numpy PyYAML opencv-python mujoco

# Ensure you have the teleop_interfaces package built in this or another sourced workspace
ccd ~/so_teleik_ws
colcon build --packages-select teleop_intrfaces

# Build the workspace
cd ~/so_teleik_ws
colcon build --packages-select so100mjk



# Source the workspace in every new terminal before running
source install/setup.bash
```

## Usage


### 1. Run Workspace Calibration (Optional but Recommended)

This script helps define the reachable workspace of the simulated arms, which is used for scaling teleoperation input. Must do if you choose to use any arm other than SO_100s, or have a different bot configuration. 

```bash
# Navigate to your workspace source directory containing calibrate_workspace.py
cd ~/so_teleik_ws/assets_calib # Adjust path if needed

# Run the calibration script
# Replace scene_triple.xml if your main scene file is named differently
# arm.yaml is the output file defining robot limits
python calibrate_workspace.py --mjcf=~/mujoco_menagerie/trs_so_arm100/scene_triple.xml --calib=config/arm.yaml

# --> Move the arms around in the MuJoCo window to explore their limits

# --> Close the window to save the arm.yaml file to the config directory
```
### 2. Run the Main Sim
```bash
# Source your workspace
source ~/so_teleik_ws/install/setup.bash

# Run the IK simulation node
ros2 run so100mjk ik_final
```

## Parameter tuning and performance enhancement
* **note** : the rosbag within /src/so100mjk/bag_files contain some rosbag examples to check. 

Depending on the behaviour there are 2 controller gain( `kp` and `kd` ) and 1 ik paramter ( `damping` ) that can be tuned to reduce the error values and improve performance. 

## Media 

![SoArm](images/Screenshot%20from%202025-10-20%2014-27-31.png)

## click here  for the demo run [video](https://drive.google.com/file/d/1KQ6kOmyYGHjQ4I7htbnXhBT-arDoGlmP/view?usp=sharing)



## License

This project is licensed under the GNU-3.0 License - see the [LICENSE](LICENSE)





