import rclpy
from rclpy.node import Node
from teleop_interfaces.msg import TeleopDataRaw
from geometry_msgs.msg import Pose 

from mujoco.glfw import glfw
import mujoco

import os
import time 
import numpy as np
import threading
from threading import Lock

'''---------------------- COUPLED DUAL ARM LM- IK SOLVER ----------------------'''
#to do later : create a seperate python helper file for this for easier usability

class CoupledDualArmIK:
    """Coupled IK solver for dual arms - solves both arms simultaneously"""
    
    def __init__(self, model, step_size, tol, damping, joint_ids, use_orientation=True):
        self.model = model
        self.step_size = step_size
        self.tol = tol
        self.damping_init = damping
        self.damping = damping
        self.use_orientation = use_orientation
        self.joint_ids = joint_ids
        
        # Jacobians
        self.jacp1 = np.zeros((3, model.nv))
        self.jacr1 = np.zeros((3, model.nv))
        self.jacp2 = np.zeros((3, model.nv))
        self.jacr2 = np.zeros((3, model.nv))
        self.jacp3 = np.zeros((3, model.nv))
        self.jacr3 = np.zeros((3, model.nv))
        
        # Quaternion helpers
        self.site_quat1 = np.zeros(4)
        self.site_quat2 = np.zeros(4)
        self.site_quat3 = np.zeros(4)
        self.error_quat1 = np.zeros(4)
        self.error_quat2 = np.zeros(4)
        self.error_quat3 = np.zeros(4)
        self.error_ori1 = np.zeros(3)
        self.error_ori2 = np.zeros(3)
        self.error_ori3 = np.zeros(3)
        
        # Errors
        self.error_pos1 = np.zeros(3)
        self.error_pos2 = np.zeros(3)
        self.error_pos3 = np.zeros(3)

    def check_joint_limits(self, q):
        """Check if all joints are within limits"""
        for i, joint_id in enumerate(self.joint_ids):
            q[i] = np.clip(q[i], 
                          self.model.jnt_range[joint_id][0], 
                          self.model.jnt_range[joint_id][1])
        return q

    def step(self, data, goal1, goal2, goal3, site_id1, site_id2, site_id3, current_q):
        """
        Single IK step - returns updated joint angles
        This is for real-time tracking (single iteration)
        """
        q = current_q.copy()
        
        # Set joint configuration
        for i, joint_id in enumerate(self.joint_ids):
            data.qpos[joint_id] = q[i]
        mujoco.mj_forward(self.model, data)
        
        # Get current end-effector poses
        current_pose1 = data.site(site_id1).xpos.copy()
        current_pose2 = data.site(site_id2).xpos.copy()
        current_pose3 = data.site(site_id3).xpos.copy()
        
        # Calculate position errors
        self.error_pos1[:] = goal1[0:3] - current_pose1
        self.error_pos2[:] = goal2[0:3] - current_pose2
        self.error_pos3[:] = goal3[0:3] - current_pose3
        
        error1_norm = np.linalg.norm(self.error_pos1)
        error2_norm = np.linalg.norm(self.error_pos2)
        error3_norm = np.linalg.norm(self.error_pos2)
        
        # Calculate Jacobians for both sites
        mujoco.mj_jacSite(self.model, data, self.jacp1, self.jacr1, site_id1)
        mujoco.mj_jacSite(self.model, data, self.jacp2, self.jacr2, site_id2)
        mujoco.mj_jacSite(self.model, data, self.jacp3, self.jacr3, site_id3)
        
        joint_ids_arm1 = self.joint_ids[0:6]
        joint_ids_arm2 = self.joint_ids[6:12]
        joint_ids_arm3 = self.joint_ids[12:18]

        jacp1_arm = self.jacp1[:, joint_ids_arm1]  # 3×6
        jacp2_arm = self.jacp2[:, joint_ids_arm2]  # 3×6
        jacp3_arm = self.jacp3[:, joint_ids_arm3]  # 3×6
        
        if self.use_orientation:
            # Calculate orientation errors
            mujoco.mju_mat2Quat(self.site_quat1, data.site(site_id1).xmat)
            mujoco.mju_negQuat(self.site_quat1, self.site_quat1)
            mujoco.mju_mulQuat(self.error_quat1, goal1[3:7], self.site_quat1)
            mujoco.mju_quat2Vel(self.error_ori1, self.error_quat1, 1.0)
            
            mujoco.mju_mat2Quat(self.site_quat2, data.site(site_id2).xmat)
            mujoco.mju_negQuat(self.site_quat2, self.site_quat2)
            mujoco.mju_mulQuat(self.error_quat2, goal2[3:7], self.site_quat2)
            mujoco.mju_quat2Vel(self.error_ori2, self.error_quat2, 1.0)

            mujoco.mju_mat2Quat(self.site_quat3, data.site(site_id3).xmat)
            mujoco.mju_negQuat(self.site_quat3, self.site_quat3)
            mujoco.mju_mulQuat(self.error_quat3, goal3[3:7], self.site_quat3)
            mujoco.mju_quat2Vel(self.error_ori3, self.error_quat3, 1.0)

            jacr1_arm = self.jacr1[:, joint_ids_arm1]  # 3x6
            jacr2_arm = self.jacr2[:, joint_ids_arm2]  # 3x6
            jacr3_arm = self.jacr3[:, joint_ids_arm3]  # 3x6

            # Combine Jacobians for each arm individually
            J1 = np.vstack([jacp1_arm, jacr1_arm])  # 6x6
            J2 = np.vstack([jacp2_arm, jacr2_arm])  # 6x6
            J3 = np.vstack([jacp3_arm, jacr3_arm])  # 6x6

            # Assemble the full (18, 18) block-diagonal Jacobian
            J = np.zeros((18, 18))
            J[0:6, 0:6] = J1
            J[6:12, 6:12] = J2
            J[12:18, 12:18] = J3

            # Stack errors (this was already correct)
            error = np.hstack([
                self.error_pos1, self.error_ori1,
                self.error_pos2, self.error_ori2,
                self.error_pos3, self.error_ori3
            ])  # Final: 18

        else:
            # Position only (9x18 Jacobian)
            J = np.zeros((9, 18))
            J[0:3, 0:6] = jacp1_arm
            J[3:6, 6:12] = jacp2_arm
            J[6:9, 12:18] = jacp3_arm

            error = np.hstack([
                self.error_pos1,
                self.error_pos2,
                self.error_pos3
            ])

        # Levenberg-Marquardt:
        n = len(self.joint_ids)
        I = np.identity(n)
        H = J.T @ J + self.damping * I
        
        try:
            delta_q = np.linalg.solve(H, J.T @ error)
        except np.linalg.LinAlgError:
            delta_q = np.linalg.lstsq(H, J.T @ error, rcond=None)[0]
        
        # Limit step size
        delta_q_norm = np.linalg.norm(delta_q)
        max_delta = 0.5
        if delta_q_norm > max_delta:
            delta_q = delta_q * (max_delta / delta_q_norm)
        
        # Update all joints
        q += self.step_size * delta_q
        
        # Apply joint limits
        q = self.check_joint_limits(q)
        
        return q, error1_norm, error2_norm, error3_norm

'''---------------------- MUJOCO Physics SIMULATOR NODE ----------------------'''

class MuJoCoSimulatorNode(Node):
    """ROS 2 Node for MuJoCo simulator with coupled IK tracking"""
    
    def __init__(self, damping=0.1):
        super().__init__("mujoco_simulator")
        self.get_logger().info("Initializing MuJoCo Simulator Node")
        
        # Load model
        menagerie_path = os.path.expanduser('~/mujoco_menagerie')
        model_path = os.path.join(menagerie_path, 'trs_so_arm100', 'scene_triple.xml')
        
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise
        
        self.joint_ids = list(range(0, 18))
        self.site_id1 = self.model.site("1attachment").id
        self.site_id2 = self.model.site("2attachment").id
        self.site_id3 = self.model.site("3attachment").id
        
        # Set initial positions
        self.data.qpos[0:6] = self.model.key_qpos[self.model.key("1home").id][0:6]
        self.data.qpos[6:12] = self.model.key_qpos[self.model.key("2home").id][0:6]
        self.data.qpos[12:18] = self.model.key_qpos[self.model.key("3home").id][0:6]
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial EE orientations for reachable goals
        ee1_quat_init = np.zeros(4)
        ee2_quat_init = np.zeros(4)
        ee3_quat_init = np.zeros(4)
        mujoco.mju_mat2Quat(ee1_quat_init, self.data.site("1attachment").xmat)
        mujoco.mju_mat2Quat(ee2_quat_init, self.data.site("2attachment").xmat)
        mujoco.mju_mat2Quat(ee3_quat_init, self.data.site("3attachment").xmat)

        self.get_logger().info(f"Initial Arm 1 EE: {self.data.site('1attachment').xpos}")
        self.get_logger().info(f"Initial Arm 2 EE: {self.data.site('2attachment').xpos}")
        self.get_logger().info(f"Initial Arm 3 EE: {self.data.site('3attachment').xpos}")

        # Default goals (will be updated from teleop)
        self.l_wrist_goal = np.array([-0.1, -0.1, 0.2, 
                                      ee1_quat_init[0], ee1_quat_init[1], 
                                      ee1_quat_init[2], ee1_quat_init[3]])
        self.r_wrist_goal = np.array([0.3, -0.1, 0.2, 
                                      ee2_quat_init[0], ee2_quat_init[1], 
                                      ee2_quat_init[2], ee2_quat_init[3]])
        self.head_goal = np.array([0.1, 0.3, 0.2, 
                                   ee3_quat_init[0], ee3_quat_init[1], 
                                   ee3_quat_init[2], ee3_quat_init[3]])
        
        # Create IK tracker
        self.ik_tracker = CoupledDualArmIK(
            self.model, step_size=1.0, tol=1e-3, 
            damping=damping, joint_ids=self.joint_ids, use_orientation=True
        )
        
        # Thread safety
        self.data_lock = Lock()
        self.sim_running = True
        self.frame_count = 0
        
        # Visualization setup
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
        
        # Subscribe to teleop data
        self.sub_teleop = self.create_subscription(
            TeleopDataRaw, '/meta/teleop_data_raw', self.teleop_callback, 10
        )
        self.get_logger().info("Subscribed to /meta/teleop_data_raw")
        
        # Start simulation thread
        self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.sim_thread.start()
        
        self.get_logger().info("MuJoCo Simulator Node initialized")
    
    def teleop_callback(self, msg: TeleopDataRaw):
        """Extract wrist poses from teleop data and update goals"""
        try:
            for group in msg.cartesian_groups:
                if group.group_name == "left_wrist":
                    self.l_wrist_goal = np.array([
                        group.pose.position.x, group.pose.position.y, group.pose.position.z + 0.8,
                        group.pose.orientation.w, group.pose.orientation.x,
                        group.pose.orientation.y, group.pose.orientation.z
                    ])
                    self.get_logger().debug(f"Left wrist goal updated: {self.l_wrist_goal[0:3]}")
                
                elif group.group_name == "right_wrist":
                    self.r_wrist_goal = np.array([
                        group.pose.position.x, group.pose.position.y, group.pose.position.z + 0.8,
                        group.pose.orientation.w, group.pose.orientation.x,
                        group.pose.orientation.y, group.pose.orientation.z
                    ])
                    self.get_logger().debug(f"Right wrist goal updated: {self.r_wrist_goal[0:3]}")
                
                elif group.group_name == "head":
                    self.head_goal = np.array([
                        group.pose.position.x, group.pose.position.y, group.pose.position.z + 0.8,
                        group.pose.orientation.w, group.pose.orientation.x,
                        group.pose.orientation.y, group.pose.orientation.z
                    ])
                    self.get_logger().debug(f"Head goal updated: {self.head_goal[0:3]}")

        except Exception as e:
            self.get_logger().error(f"Teleop callback error: {e}")
    
    def simulation_loop(self):
        """Runs physics with IK tracking"""
        self.get_logger().info("Simulation loop started")
        
        while self.sim_running and rclpy.ok():
            with self.data_lock:
                # Get current joint configuration
                current_q = np.array([self.data.qpos[i] for i in self.joint_ids])
                
                # Compute one IK step for both arms (coupled)
                q_target, err1, err2, err3 = self.ik_tracker.step(
                self.data, self.l_wrist_goal, self.r_wrist_goal, self.head_goal,
                self.site_id1, self.site_id2, self.site_id3, current_q
                )
                
                # Apply to controller
                self.data.ctrl[:18] = q_target
                
                # Step physics
                mujoco.mj_step(self.model, self.data)
                
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    ee1 = self.data.site("1attachment").xpos
                    ee2 = self.data.site("2attachment").xpos
                    ee3 = self.data.site("3attachment").xpos
                    self.get_logger().info(
                        f"Frame {self.frame_count}: "
                        f"EE1=[{ee1[0]:.3f}, {ee1[1]:.3f}, {ee1[2]:.3f}] "
                        f"Goal1=[{self.l_wrist_goal[0]:.3f}, {self.l_wrist_goal[1]:.3f}, {self.l_wrist_goal[2]:.3f}] "
                        f"Err1={err1:.4f} | "
                        f"EE2=[{ee2[0]:.3f}, {ee2[1]:.3f}, {ee2[2]:.3f}] "
                        f"Goal2=[{self.r_wrist_goal[0]:.3f}, {self.r_wrist_goal[1]:.3f}, {self.r_wrist_goal[2]:.3f}] "
                        f"Err2={err2:.4f} | "
                        f"EE3=[{ee3[0]:.3f}, {ee3[1]:.3f}, {ee3[2]:.3f}] "
                        f"Goal3=[{self.head_goal[0]:.3f}, {self.head_goal[1]:.3f}, {self.head_goal[2]:.3f}] "
                        f"Err3={err3:.4f}"
                    )
            
            time.sleep(0.001)  # ~1kHz
    '''---------------------- VISUALISER & support fucnctions----------------------'''
    def visualization_loop(self):
        """Run visualization (blocking, main thread)"""
        self.get_logger().info("Starting visualization")
        
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Dual Arm IK Tracker", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
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
        
        while not glfw.window_should_close(self.window) and self.sim_running:
            width, height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            
            with self.data_lock:
                mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, 
                                      mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            
            mujoco.mjr_render(viewport, self.scene, self.mjr_context)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
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
        self.sim_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoSimulatorNode()
    
    # Spin in a thread so callbacks are processed
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        node.visualization_loop()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()