import rclpy
from rclpy.node import Node
from teleop_interfaces.msg import TeleopDataRaw
from geometry_msgs.msg import Pose

from ament_index_python.packages import get_package_share_directory
import yaml

from mujoco.glfw import glfw
import mujoco

import os
import time 
import numpy as np
import threading
from threading import Lock
import cv2

'''---------------------- SINGLE ARM LM-IK SOLVER ----------------------'''
class SingleArmIK:
    def __init__(self, model, damping, joint_ids, site_name, use_orientation=True):
        self.model = model
        self.damping = damping
        self.joint_ids = joint_ids
        self.site_id = model.site(site_name).id
        self.use_orientation = use_orientation
        
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.site_quat = np.zeros(4)
        self.error_quat = np.zeros(4)

    def step(self, data, goal_pose, current_q):
        current_ee_pos = data.site(self.site_id).xpos
        error_pos = goal_pose[0:3] - current_ee_pos
        error_norm = np.linalg.norm(error_pos)

        jacp_arm = self.jacp[:, self.joint_ids]
        
        if self.use_orientation:
            mujoco.mju_mat2Quat(self.site_quat, data.site(self.site_id).xmat)
            mujoco.mju_negQuat(self.site_quat, self.site_quat)
            mujoco.mju_mulQuat(self.error_quat, goal_pose[3:7], self.site_quat)
            error_ori = np.zeros(3)
            mujoco.mju_quat2Vel(error_ori, self.error_quat, 1.0)
            
            jacr_arm = self.jacr[:, self.joint_ids]
            
            J = np.vstack([jacp_arm, jacr_arm])
            error = np.hstack([error_pos, error_ori])
        else:
            J = jacp_arm
            error = error_pos

        I = np.identity(len(self.joint_ids))
        H = J.T @ J + self.damping * I
        
        try:
            delta_q = np.linalg.solve(H, J.T @ error)
        except np.linalg.LinAlgError:
            delta_q = np.linalg.lstsq(H, J.T @ error, rcond=None)[0]

        q_target = current_q + delta_q
        for i, joint_idx in enumerate(self.joint_ids):
            q_target[i] = np.clip(q_target[i], 
                                  self.model.jnt_range[joint_idx][0], 
                                  self.model.jnt_range[joint_idx][1])
            
        return q_target, error_norm, current_ee_pos, goal_pose[0:3]

'''---------------------- MUJOCO Physics SIMULATOR NODE ----------------------'''

class MuJoCoSimulatorNode(Node):
    def __init__(self, damping=0.1):
        super().__init__("mujoco_simulator")
        self.get_logger().info("Initializing MuJoCo Simulator Node")
        
        # --- Model Loading ---
        menagerie_path = os.path.expanduser('~/mujoco_menagerie')
        model_path = os.path.join(menagerie_path, 'trs_so_arm100', 'scene_triple.xml')
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        # --- Calibration Loading ---
        self.calibrate_data = None
        self.arm_data = None
        self.package_path = get_package_share_directory('so100mjk')
        self.calibration_file = os.path.join(self.package_path, 'config', 'calibrate.yaml')
        self.arm_file = os.path.join(self.package_path, 'config', 'arm.yaml')
        self.multiplication_factors = {}
        self.arm_min = {"left_wrist": {"x_offset": 0.0, "y_offset": 0.0, "z_offset": 0.0}, "right_wrist": {"x_offset": 0.0, "y_offset": 0.0, "z_offset": 0.0}, "head": {"x_offset": 0.0, "y_offset": 0.0, "z_offset": 0.0}}
        self.calibrate_min = {"left_wrist": {"x_offset": 0.0, "y_offset": 0.0, "z_offset": 0.0}, "right_wrist": {"x_offset": 0.0, "y_offset": 0.0, "z_offset": 0.0}, "head": {"x_offset": 0.0, "y_offset": 0.0, "z_offset": 0.0}}
        self.calibrate_helper()

        # --- Visualization Setup ---
        self.cam = None
        self.opt = None
        self.scene = None
        self.mjr_context = None
        self.window = None
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        
        # --- IK Solvers ---
        self.ik_solvers = [
            SingleArmIK(self.model, damping, list(range(0, 6)), "1attachment"),
            SingleArmIK(self.model, damping, list(range(6, 12)), "2attachment"),
            SingleArmIK(self.model, damping, list(range(12, 18)), "3attachment")
        ]
        
        # --- Initial State & Goals ---
        self.data.qpos[0:6] = self.model.key_qpos[self.model.key("1home").id][0:6]
        self.data.qpos[6:12] = self.model.key_qpos[self.model.key("2home").id][0:6]
        self.data.qpos[12:18] = self.model.key_qpos[self.model.key("3home").id][0:6]
        mujoco.mj_forward(self.model, self.data)
        
        self.goals = []
        for solver in self.ik_solvers:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.site(solver.site_id).xmat)
            pos = self.data.site(solver.site_id).xpos
            self.goals.append(np.hstack([pos, quat]))

        self.get_logger().info(f"Initial Arm 1 EE: {self.goals[0][:3]}")
        self.get_logger().info(f"Initial Arm 2 EE: {self.goals[1][:3]}")
        self.get_logger().info(f"Initial Arm 3 EE: {self.goals[2][:3]}")

        # --- CAMERA SETUP (PART 1: Initialize but DON'T render yet) ---
        self.cv_window_name = "MuJoCo Camera View"
        self.cam_height = 600
        self.cam_width = 400
        self.camera_names = ["1wrist_cam", "2wrist_cam", "3wrist_cam"]
        self.renderers = {}  # EMPTY FOR NOW - will be filled after OpenGL context is ready
        self.cameras_ready = False  # Flag to track if cameras are initialized

        # --- ROS & Threading Setup ---
        self.data_lock = Lock()
        self.sim_running = True
        self.frame_count = 0

        self.sub_teleop = self.create_subscription(
            TeleopDataRaw, '/meta/teleop_data_raw', self.teleop_callback, 10
        )
        self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.sim_thread.start()
        
        self.get_logger().info("MuJoCo Simulator Node initialized")

    def calibrate_helper(self):
        try:
            with open(self.calibration_file, 'r') as f:
                self.calibrate_data = yaml.safe_load(f)['calibrate']
            if not self.calibrate_data:
                self.get_logger().error("Calibration YAML is empty or malformed.")
                return False
            self.get_logger().info("Human calibration data loaded successfully.")
        except (FileNotFoundError, KeyError) as e:
            self.get_logger().error(f"Failed to load human data: {e}")
            return False

        try:
            with open(self.arm_file, 'r') as f:
                self.arm_data = yaml.safe_load(f)['calibrate']
            if not self.arm_data:
                self.get_logger().error("Arm data YAML is empty or malformed.")
                return False
            self.get_logger().info("Robot arm data loaded successfully.")
        except (FileNotFoundError, KeyError) as e:
            self.get_logger().error(f"Failed to load arm data: {e}")
            return False

        part_names = ['left_wrist', 'right_wrist', 'head']

        for name in part_names:
            try:
                human_part = self.calibrate_data[name]
                robot_part = self.arm_data[name]

                human_x_range = human_part['x_offset']['max'] - human_part['x_offset']['min']
                human_y_range = human_part['y_offset']['max'] - human_part['y_offset']['min']
                human_z_range = human_part['z_offset']['max'] - human_part['z_offset']['min']

                robot_x_range = robot_part['x_offset']['max'] - robot_part['x_offset']['min']
                robot_y_range = robot_part['y_offset']['max'] - robot_part['y_offset']['min']
                robot_z_range = robot_part['z_offset']['max'] - robot_part['z_offset']['min']

                x_factor = robot_x_range / human_x_range if human_x_range != 0 else 1.0
                y_factor = robot_y_range / human_y_range if human_y_range != 0 else 1.0
                z_factor = robot_z_range / human_z_range if human_z_range != 0 else 1.0
                
                self.multiplication_factors[name] = {
                    "x_factor": x_factor,
                    "y_factor": y_factor,
                    "z_factor": z_factor,
                }

                self.calibrate_min[name] = {
                    "x_offset": human_part['x_offset']['min'],
                    "y_offset": human_part['y_offset']['min'],
                    "z_offset": human_part['z_offset']['min'],
                }

                self.arm_min[name] = {
                    "x_offset": robot_part['x_offset']['min'],
                    "y_offset": robot_part['y_offset']['min'],
                    "z_offset": robot_part['z_offset']['min'],
                }
            except KeyError as e:
                self.get_logger().error(f"Missing key {e} for part '{name}' in a config file.")
                return False
        
        self.get_logger().info(f"Calculated scaling factors: {self.multiplication_factors}")

    def teleop_callback(self, msg: TeleopDataRaw):
        """Update goals from teleop data."""
        group_map = {"left_wrist": 0, "right_wrist": 1, "head": 2}
        for group in msg.cartesian_groups:
            if group.group_name in group_map:
                idx = group_map[group.group_name]
                self.goals[idx] = np.array([
                    (group.pose.position.x  - self.calibrate_min[group.group_name]["x_offset"])*self.multiplication_factors[group.group_name]["x_factor"] + self.arm_min[group.group_name]["x_offset"],
                    (group.pose.position.y  - self.calibrate_min[group.group_name]["y_offset"])*self.multiplication_factors[group.group_name]["y_factor"] + self.arm_min[group.group_name]["y_offset"],
                    (group.pose.position.z  - self.calibrate_min[group.group_name]["z_offset"])*self.multiplication_factors[group.group_name]["z_factor"] + self.arm_min[group.group_name]["z_offset"],
                    group.pose.orientation.w, group.pose.orientation.x,
                    group.pose.orientation.y, group.pose.orientation.z
                ])
    
    def init_camera_renderers(self):
        """
        Initialize camera renderers AFTER OpenGL context is ready.
        This should be called from visualization_loop() after GLFW window is created.
        """
        try:
            for cam_name in self.camera_names:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cam_id == -1:
                    raise ValueError(f"Camera '{cam_name}' not found in model.")

                self.get_logger().info(f"Setting up renderer for camera '{cam_name}' (ID: {cam_id})")

                cam_render = mujoco.MjvCamera()
                cam_render.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam_render.fixedcamid = cam_id

                scene_render = mujoco.MjvScene(self.model, maxgeom=1000)
                context_render = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
                viewport_render = mujoco.MjrRect(0, 0, self.cam_width, self.cam_height)
                rgb_buffer = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)
                depth_buffer = np.zeros((self.cam_height, self.cam_width, 1), dtype=np.float32)

                self.renderers[cam_name] = {
                    "id": cam_id,
                    "cam": cam_render,
                    "scn": scene_render,
                    "con": context_render,
                    "vp": viewport_render,
                    "rgb": rgb_buffer,
                    "dep": depth_buffer
                }

            self.cameras_ready = True
            self.get_logger().info("Camera renderers initialized successfully")

        except Exception as e:
            self.get_logger().error(f"Error initializing camera renderers: {e}")
            self.renderers = {}
            self.cameras_ready = False

    def cv_frame_update(self):
        """
        Renders camera views and displays them in a tiled OpenCV window.
        Layout: Top row = head camera (full width)
                Bottom row = left wrist (left half) | right wrist (right half)
        """
        if not self.cameras_ready or not self.renderers:
            if self.frame_count == 1:  # Log once at startup
                self.get_logger().warn(f"Cameras not ready. cameras_ready={self.cameras_ready}, renderers_empty={not self.renderers}")
            return

        images = {}

        # --- Render each camera view ---
        for cam_name, r in self.renderers.items():
            try:
                # Update scene with current data, using the specific camera
                # Use data_lock to ensure data isn't being modified by sim thread
                with self.data_lock:
                    mujoco.mjv_updateScene(
                        self.model, self.data, self.opt, None,
                        r["cam"], mujoco.mjtCatBit.mjCAT_ALL, r["scn"]
                    )

                # Render scene to offscreen buffer
                mujoco.mjr_render(r["vp"], r["scn"], r["con"])

                # Read RGB and depth pixels from OpenGL framebuffer
                mujoco.mjr_readPixels(r["rgb"], r["dep"], r["vp"], r["con"])

                # Process for OpenCV
                img_flipped = np.flipud(r["rgb"])  # Flip vertical (OpenGL origin is bottom-left)
                img_bgr = cv2.cvtColor(img_flipped, cv2.COLOR_RGB2BGR)  # RGB -> BGR

                img_rot = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)  # Rotate if needed
                images[cam_name] = img_rot

                # Debug: Check pixel values (only log once)
                if self.frame_count == 1 and cam_name == "1wrist_cam":
                    unique_vals = np.unique(r["rgb"])
                    if len(unique_vals) > 1:
                        self.get_logger().info(f"Camera {cam_name}: pixel value range = {unique_vals[:5]}...{unique_vals[-5:]}")
                    else:
                        self.get_logger().info(f"Camera {cam_name}: pixel value = {unique_vals}")


            except Exception as e:
                self.get_logger().warn(f"Failed to render camera {cam_name}: {e}")
                return

        # --- Create the tiled layout ---
        img_head = images.get("3wrist_cam")
        img_left = images.get("2wrist_cam")
        img_right = images.get("1wrist_cam")

        if img_head is None or img_left is None or img_right is None:
            # Not all images were rendered, wait for next frame
            return

        # Get dimensions from head image
        h_head, w_head, _ = img_head.shape
        if h_head == 0 or w_head == 0:
            return # Skip if image is empty

        # Resize left and right images to half the width and same height as head
        target_w = w_head // 2
        target_h = h_head
        img_left_resized = cv2.resize(img_left, (target_w, target_h))
        img_right_resized = cv2.resize(img_right, (target_w, target_h))

        # Create bottom row by horizontally stacking left and right
        bottom_row = np.hstack((img_left_resized, img_right_resized))

        # Resize head image to match bottom row width
        head_width = bottom_row.shape[1]
        head_height = bottom_row.shape[0]
        img_head_resized = cv2.resize(img_head, (head_width, head_height))

        # Create final image by vertically stacking head (top) and bottom row
        final_image = np.vstack((img_head_resized, bottom_row))

        # Display the combined image
        cv2.imshow(self.cv_window_name, final_image)
        cv2.waitKey(1)

    def simulation_loop(self):
        """Runs physics with parallel IK tracking."""
        self.get_logger().info("Simulation loop started")
        q_target = np.zeros(self.model.nu)
        
        # Wait for cameras to initialize before starting rendering calls
        startup_timeout = time.time() + 5.0  # 5 second timeout
        while not self.cameras_ready and time.time() < startup_timeout:
            time.sleep(0.1)
        
        if self.cameras_ready:
            self.get_logger().info("Cameras ready, sim loop proceeding")
        else:
            self.get_logger().warn("Cameras not ready after timeout, sim loop proceeding")
        
        while self.sim_running and rclpy.ok():
            with self.data_lock:
                mujoco.mj_forward(self.model, self.data)
                for solver in self.ik_solvers:
                    mujoco.mj_jacSite(self.model, self.data, solver.jacp, solver.jacr, solver.site_id)
                
                threads = []
                results = [None] * 3
                
                def solve_ik_for_arm(arm_idx):
                    solver = self.ik_solvers[arm_idx]
                    current_q_arm = self.data.qpos[solver.joint_ids]
                    goal_pose = self.goals[arm_idx]
                    results[arm_idx] = solver.step(self.data, goal_pose, current_q_arm)

                for i in range(3):
                    thread = threading.Thread(target=solve_ik_for_arm, args=(i,))
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                q_targets_list, errors_list, current_ee_pos_list, goal_ee_pos_list = zip(*results)
                q_target = np.concatenate(q_targets_list)

                kp = 100
                kd = 5
                current_qpos = self.data.qpos[:18]
                current_qvel = self.data.qvel[:18]

                pos_error = q_target - current_qpos
                torque = kp * pos_error - kd * current_qvel
                self.data.ctrl[:18] = torque

                # --- THIS CALL HAS BEEN MOVED TO visualization_loop() ---
                # self.cv_frame_update() 
                # --------------------------------------------------------

                mujoco.mj_step(self.model, self.data)
                
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    def format_vec(vec):
                        return f"[{vec[0]: 6.3f}, {vec[1]: 6.3f}, {vec[2]: 6.3f}]"

                    self.get_logger().info(f"--- Frame {self.frame_count} ---")
                    self.get_logger().info(
                        f"  Arm1: PosErr={errors_list[0]:.4f} | "
                        f"Pos={format_vec(current_ee_pos_list[0])} | "
                        f"Goal={format_vec(goal_ee_pos_list[0])}"
                    )
                    self.get_logger().info(
                        f"  Arm2: PosErr={errors_list[1]:.4f} | "
                        f"Pos={format_vec(current_ee_pos_list[1])} | "
                        f"Goal={format_vec(goal_ee_pos_list[1])}"
                    )
                    self.get_logger().info(
                        f"  Arm3: PosErr={errors_list[2]:.4f} | "
                        f"Pos={format_vec(current_ee_pos_list[2])} | "
                        f"Goal={format_vec(goal_ee_pos_list[2])}"
                    )
            time.sleep(0.001)

    def visualization_loop(self):
        """Run visualization (blocking, main thread)"""
        self.get_logger().info("Starting visualization")
        
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Decoupled Arm IK Tracker", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # Initialize rendering options FIRST before camera rendering
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.mjr_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 90, -30, 1.5
        self.cam.lookat = np.array([0.0, 0.2, 0.3])
        
        glfw.set_key_callback(self.window, self.keyboard_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move_callback)

        self.init_camera_renderers()
        
        while not glfw.window_should_close(self.window) and self.sim_running:
            width, height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            
            with self.data_lock:
                mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, 
                                      mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            
            mujoco.mjr_render(viewport, self.scene, self.mjr_context)

            # --- RENDER CAMERA FRAMES HERE (using main GL context) ---
            self.cv_frame_update()
            # ----------------------------------------------------------

            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        self.get_logger().info("Visualization loop ending.")
        self.sim_running = False # Signal sim thread to stop
        glfw.terminate()
    
    def keyboard_callback(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            with self.data_lock:
                mujoco.mj_resetData(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
            self.get_logger().info("Simulation reset")

    def scroll_callback(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, 
                            self.scene, self.cam)
    
    def mouse_button_callback(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        self.lastx, self.lasty = glfw.get_cursor_pos(window)
    
    def mouse_move_callback(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx, self.lasty = xpos, ypos
        
        if not (self.button_left or self.button_middle or self.button_right):
            return
        
        width, height = glfw.get_window_size(window)
        mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or 
                     glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        
        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        
        mujoco.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)
    
    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.sim_running = False
        if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
            self.get_logger().info("Waiting for simulation thread to join...")
            self.sim_thread.join(timeout=2.0)
            if self.sim_thread.is_alive():
                self.get_logger().warn("Simulation thread did not join in time.")
        cv2.destroyAllWindows()
        self.get_logger().info("Node destroyed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoSimulatorNode()
    
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        node.visualization_loop()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    except Exception as e:
        node.get_logger().error(f"Error in visualization loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.get_logger().info("Main loop finished, shutting down ROS...")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        
        # Wait for the spin thread to finish
        if spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
        
        node.get_logger().info("Shutdown complete.")


if __name__ == "__main__":
    main()