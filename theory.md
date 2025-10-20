
# Project Theory

This document outlines the theoretical concepts behind the key components implemented in the SoTeleIK project.

## Robotic Arm [Resource](https://youtu.be/P_PP76flZfw?si=uuJcPit-Vl1YqpIM)

The simulation features three **so_100** robotic arms. Each arm is described as having **5 + 1 Degrees of Freedom (DoF)**.

* **Degrees of Freedom (DoF):** This refers to the number of independent ways a rigid body or a mechanical system can move. For a robotic arm, it typically corresponds to the number of joints that can be independently controlled.
* **5 + 1 DoF:** This likely means the arm has 5 main revolute joints controlling its pose (position and orientation) in 3D space, plus 1 additional joint for the gripper (jaw) actuation, making a total of 6 independently controlled joints. The first 5 DoF determine the end-effector's reach and orientation, while the last DoF controls grasping.

---

## Inverse Kinematics (IK) Theory : [Resource](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
**Inverse Kinematics (IK)** is the process of calculating the required **joint parameters** (like angles for revolute joints) of a robotic arm so that its **end-effector** reaches a desired **pose** (position and orientation) in the workspace. 

* **Forward Kinematics vs. Inverse Kinematics:** Forward kinematics calculates the end-effector pose given the joint angles. Inverse kinematics does the opposite.
* **Levenberg-Marquardt (LM) Algorithm:** This project uses the LM algorithm for IK. LM is an iterative numerical optimization method used to solve non-linear least squares problems.
    * It's well-suited for IK because it effectively finds joint angles that minimize the error between the current end-effector pose and the desired goal pose.
    * It blends the **Gauss-Newton algorithm** and the method of **gradient descent**. It acts more like gradient descent when far from the solution (ensuring progress) and more like Gauss-Newton when close to the solution (for faster convergence).
    * It generally handles singularities (arm configurations where movement is restricted) and redundancy (more joints than needed for a task) more robustly than simpler methods like the Jacobian pseudoinverse.
    
* **Handling Singularities with LM:**
    * A **singularity** is a robot configuration (often at workspace boundaries or when joints align) where the **Jacobian matrix ($J$)** loses rank. The Jacobian relates joint velocities to end-effector velocity. Near a singularity, inverting the Jacobian ($J^{-1}$) or related matrices ($J^T J$) becomes numerically unstable, involving division by very small numbers.
    * Simple methods like using the Jacobian (pseudo)inverse can command extremely large joint velocities ($\Delta q$) near singularities, causing instability.
    * LM avoids this instability by introducing a **damping factor ($\lambda$)**. It solves the equation $(J^T J + \lambda I) \Delta q = J^T e$, where $I$ is the identity matrix and $e$ is the pose error.
    * The added $\lambda I$ term ensures that the matrix $(J^T J + \lambda I)$ is **always well-conditioned and invertible**, even if $J^T J$ is singular. It prevents division by near-zero values.
    * When the algorithm approaches a singularity, it typically increases $\lambda$. This makes the update step smaller and behave more like **gradient descent**, sacrificing speed for stability and avoiding large, erratic joint movements. When far from a singularity, $\lambda$ is decreased, allowing faster Gauss-Newton-like convergence.
    * In essence, the damping provides **numerical robustness**, allowing the solver to navigate near singular configurations without "blowing up".

---

## MuJoCo : [Resource](https://www.youtube.com/@pranavab)

**MuJoCo (Multi-Joint dynamics with Contact)** is a high-performance physics engine specializing in simulating the dynamics of articulated rigid bodies, particularly those involving contact forces.

* **Physics Simulation:** It calculates the motion of objects over time based on physical laws (Newton-Euler dynamics), considering factors like mass, inertia, joints, actuators, and external forces (like gravity).
* **Contact Modeling:** MuJoCo uses an efficient and stable approach to model contact forces between objects, which is crucial for realistic simulation of grasping, walking, or interaction with the environment.
* **Actuation:** It allows defining various actuator types (like position servos, motors applying torque, etc.) to control the simulated robot's joints.
* **Rendering:** MuJoCo includes visualization capabilities (often used via GLFW bindings in Python) to render the simulated scene in 3D, including offscreen rendering to specific cameras defined in the model.

---

## PD Controller : [Resource](https://www.youtube.com/channel/UCq0imsn84ShAe9PBOFnoIrg)

A **Proportional-Derivative (PD) Controller** is a type of feedback control loop mechanism widely used in robotics to control the position or velocity of joints, especially when using **torque-based actuators** (like `<motor>` in MuJoCo).

* **Goal:** To drive a system (like a robot joint) towards a target setpoint (desired position, $q_{target}$) by applying a calculated control signal (torque, $\tau$).
* **Components:**
    * **Proportional (P) Term ($K_p$):** Calculates an output proportional to the current **error** ($e = q_{target} - q_{current}$). It acts like a virtual spring pulling the joint towards the target. A higher $K_p$ results in a stronger pull (stiffer spring) but can lead to overshoot and oscillations.
        $\tau_P = K_p \cdot e$
    * **Derivative (D) Term ($K_d$):** Calculates an output proportional to the **rate of change of the error** (approximated by the negative of the current joint velocity, $-\dot{q}_{current}$, assuming target velocity is zero). It acts like a virtual damper, resisting motion and reducing oscillations. A higher $K_d$ increases damping.
        $\tau_D = -K_d \cdot \dot{q}_{current}$
* **Control Law:** The total torque applied is the sum of the P and D terms:
    $\tau = \tau_P + \tau_D = K_p \cdot (q_{target} - q_{current}) - K_d \cdot \dot{q}_{current}$
* **Tuning:** The gains $K_p$ and $K_d$ must be carefully tuned for each specific robot and task to achieve stable, responsive, and accurate tracking without excessive oscillation or sluggishness.

* For additional info **consider following my github repo [Control Manual](https://github.com/AkshayArjun/Control-Manual) where I will keep adding more of my notes in control systems** .
---

## CV Windowing : [Resource](https://opencv.org/)

This project uses **OpenCV (`cv2`)**, a popular computer vision library, to display real-time video feeds from the cameras attached to the wrists of the simulated robot arms.

* **Offscreen Rendering:** MuJoCo's rendering capabilities are used to generate images from specific cameras defined in the XML model (`<camera name="wrist_cam">`). This rendering happens "offscreen," meaning it doesn't rely on the main simulation window.
* **Pixel Extraction:** The rendered image data (RGB pixel values) is read from MuJoCo's rendering context into a NumPy array.
* **Image Processing:**
    * The image is vertically flipped (`np.flipud`) because MuJoCo's OpenGL rendering origin (bottom-left) differs from OpenCV's image origin (top-left).
    * The color format is converted from RGB (MuJoCo) to BGR (`cv2.cvtColor`), which is the standard format used by OpenCV for display.
    * Images may be resized (`cv2.resize`) and combined (`np.hstack`, `np.vstack`) to create the desired tiled layout.
* **Display:** The final processed NumPy array (representing the tiled image) is displayed in a separate window using `cv2.imshow()`. The `cv2.waitKey(1)` function is crucial to allow OpenCV to process window events and refresh the display.

---

## Python Threading : [Resource](https://www.geeksforgeeks.org/python/multithreading-python-set-1/)

**Threading** allows different parts of the Python program to run concurrently, improving responsiveness and enabling parallel tasks.

* **Concurrency:** In this project, threading is essential for running multiple loops simultaneously:
    * **ROS 2 Spin:** Listens for incoming messages (like teleop commands) and executes callbacks (`teleop_callback`).
    * **MuJoCo Simulation Loop (`simulation_loop`):** Continuously calculates IK, applies PD control torques, steps the physics simulation (`mujoco.mj_step`), and triggers camera rendering.
    * **Main Visualization Loop (`visualization_loop`):** Handles the main GLFW window rendering, user interaction (mouse/keyboard), and initializes the offscreen camera renderers.
* **`threading.Thread`:** Used to create and manage the separate threads for the simulation loop and the ROS 2 spin.
* **`threading.Lock` (`data_lock`):** A synchronization primitive used to prevent **race conditions**. When multiple threads need to access or modify shared data (like `self.data`, `self.goals`), the lock ensures that only one thread can access that data at any given time. A thread acquires the lock (`with self.data_lock:`), performs its operations on the shared data, and then releases the lock automatically, preventing other threads from interfering and causing corrupted data or crashes.