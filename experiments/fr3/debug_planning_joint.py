import os
import threading
import time
import copy
from pprint import pformat
import argparse
from math import sin, cos

from hydrax.algs import MPPI, MTP, AnMTP
from hydrax.utils.utils import se3_left_invariant_metric
from hydrax.tasks.pusht_franka_free import PushTFranka

from hydrax.alg_base import SamplingBasedController

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R

import jax
jax.config.update("jax_platform_name", "gpu")
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx

from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Point
from franka_panda_server import FrankaPandaServer

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix
from tf2_geometry_msgs import do_transform_pose
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from pynput import keyboard

def np_yaw_from_quat(x, y, z, w):
    # standard ZYX Euler convention
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw + np.pi/2



class FR3_PushT(FrankaPandaServer):

    def __init__(self, 
                 ctrl,
                 robot_ip,
                 seed,
                 debug_model,
                 debug_data,
                 viewer,
                 trace_idxs,
                 ):
        
        super().__init__(robot_ip, 
                         None) 
        
        self.get_logger().info(
            "Initializing SMPC Controller..."
        )
        self.ctrl = ctrl
        self.debug_model = debug_model
        self.debug_data = debug_data
        self.viewer = viewer
        
        ####################################
        ##    MoveIt Safety Constraints   ##    
        ####################################
        self.add_collision_primitive(
            id="table",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(1.2, 0.9, 0.1),
            position=np.array([0.4, 0.0, -0.05]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.add_collision_primitive(
            id="wall x",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(0.05, 1.0, 0.5),
            position=np.array([1.025, 0.0, 0.15]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.add_collision_primitive(
            id="wall y_neg",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(1.2, 0.05, 0.5),
            position=np.array([0.4, -0.475, 0.15]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.add_collision_primitive(
            id="wall y_pos",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(1.2, 0.05, 0.5),
            position=np.array([0.4, 0.475, 0.15]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        ####################################
        ##       Move to initial pose     ##    
        ####################################
        
        self.init_pos = np.array([0.6, 0.2, 0.156])   # 0,26
        # self.init_pos = np.array([0.5, 0.0, 0.255])   # 0,26
        # add small noise: keep x small, increase variance in y
        self.init_pos[0] += np.random.uniform(-0.03 , 0.03)   # x
        self.init_pos[1] += np.random.uniform(-0.03, 0.03)   # y (larger variance)
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0])
       
        self.init_rot = R.from_quat(self.init_quat).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = self.init_rot
        pose[:3, 3] = self.init_pos
        self.plan_and_move_to_pose(pose)

        ####################################
        ##            T Object            ##    
        ####################################

        self.lin_t = None
        self.quat_t = None
        self.robot_q = None
        self.robot_dq = None

        # Fix, already initialized in RobotServer
        self.tf_buffer = self._tf_buffer
        self.tf_listener = self._tf_listener

        self.br = StaticTransformBroadcaster(self)
        self._publish_static_robot_tf()

        # mocap
        self.eq_id = mujoco.mj_name2id(self.ctrl.task.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, "sensor_coupling")

        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tf_buffer.can_transform("fr3_link0", "objectPushT_MuJoCo",
                                            rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=0.0)):
                break
        time.sleep(0.5)  # wait a bit more
        
        ####################################
        ##         JIT Controller         ##    
        ####################################

        # Wait until all states are received
        while not self._states_received():
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(
            "All states received, initializing controller..."
        )
        self.trace_idxs = trace_idxs
  
        id = mujoco.mj_name2id(self.ctrl.task.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ghost_block")
        self.ghost_id = self.ctrl.task.mj_model.body_mocapid[id]
        
        #! mj_model on CPU
        #! model on GPu
        # Create mjx_data on host,  push to device
        self.mjx_data = mjx.make_data(self.ctrl.task.model)
        self.policy_params = self.ctrl.init_params(seed)

        self.get_logger().info(
            f"Planning with {self.ctrl.task.planning_horizon} steps "
            f"over a {self.ctrl.task.planning_horizon * self.ctrl.task.dt} second horizon."
        )

        while self.lin_t is None or self.quat_t is None or self.robot_q is None or self.robot_dq is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.lin_t, self.quat_t = self._update_T()  
            self.robot_q, self.robot_dq = self._get_robot_state_np()
        self.get_logger().info(
            "Initial object and robot states received, starting controller..."
        )

        # Do a forward once (host side is fine here)
        self.mjx_data = mjx.forward(self.ctrl.task.model, self.mjx_data)

        # Make unified jitted step function
        self.jit_step = self.make_jitted_step(ctrl)

        self.get_logger().info("Jitting controller...")
        st = time.time()

        # # Warmstart on device with dummy data (zeros)
        # lin0 = jnp.zeros(3, dtype=jnp.float32)
        # quat0 = jnp.array([0., 0., 0., 1.], dtype=jnp.float32)
        # q0 = jnp.zeros(7, dtype=jnp.float32)
        # dq0 = jnp.zeros(7, dtype=jnp.float32)

        # One call to transfer mjx_data & policy_params to GPU and compile
        self.jit_step = self.jit_step.lower(
            self.mjx_data, self.policy_params,
            self.lin_t, self.quat_t, self.robot_q, self.robot_dq
        ).compile()
        
        # warmstart simulation
        # to resolve mocap vs object initial discrepancy
        for _ in range(20):
            self.mjx_data = mjx.step(self.ctrl.task.model, self.mjx_data)
        # warmstart controller
        for _ in range(5):
            self.mjx_data, self.policy_params, _ = self.jit_step(
                self.mjx_data, self.policy_params,
                self.lin_t, self.quat_t, self.robot_q, self.robot_dq
            )
        
        self.get_logger().info(f"Time to jit and warmstart: {time.time() - st:.3f} s")

        ####################################
        ##           Start Timer          ##    
        ####################################
        self._key_lock = threading.Lock()
        self.teleop_enabled = True 
        self._key_vx = 0.0
        self._key_vy = 0.0

        # Start keyboard listener in background
        threading.Thread(target=self._keyboard_loop, daemon=True).start()


        self.finished_task = False
        self.servo_freq = 50  # Hz
        self.plan_freq = 5
        self.action = None
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        self.sim_group = MutuallyExclusiveCallbackGroup()

        self.create_timer(1.0 / 3, self._run_controller, callback_group=self.sim_group)
        self.create_timer(1.0 / self.servo_freq, self._send_keyboard_command, callback_group=self.parallel_group)
        # time.sleep(0.2)
        # self.create_timer(1.0 / self.servo_freq, self._send_command, callback_group=self.parallel_group)
        

    def _keyboard_loop(self):
            """
            Background key listener: updates desired vx/vy based on arrow keys.
            Uses locks because pynput runs on its own thread.
            """

            def clamp(v, lo, hi):
                return max(lo, min(hi, v))

            def on_press(key):
                # Toggle teleop with 't' (optional)
                try:
                    if key.char == 't':
                        self.teleop_enabled = not self.teleop_enabled
                        self.get_logger().info(f"teleop_enabled = {self.teleop_enabled}")
                        return
                    if key.char == ' ':
                        # space = stop
                        with self._key_lock:
                            self._key_vx = 0.0
                            self._key_vy = 0.0
                        return
                except AttributeError:
                    pass

                vx, vy = 0.0, 0.0

                if key == keyboard.Key.up:
                    vx = 0.05
                elif key == keyboard.Key.down:
                    vx = -0.05
                elif key == keyboard.Key.left:
                    vy = +0.05
                elif key == keyboard.Key.right:
                    vy = -0.05
                else:
                    return
                with self._key_lock:
                    self._key_vx = vx
                    self._key_vy = vy

            def on_release(key):
                # When arrow key released -> stop (simple behavior)
                if key in (keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right):
                    with self._key_lock:
                        self._key_vx = 0.0
                        self._key_vy = 0.0

            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
    
    def _publish_static_robot_tf(self):
        """
        Publish static transforms:
            
        1. Transform from 'fr3_link0' (robot base) to 'optitrack' (motion capture system)
            - Based on calibration data

        2. Transform from 'objectPushT' (real object) to 'objectPushT_MuJoCo' (simulated object)
            - Based on difference between centre of obtitrack rigid body and mujoco body 

        Note:
            The transforms are latched by the broadcaster, so they persist after publishing.

        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'fr3_link0'
        t.child_frame_id = 'optitrack'

        t.transform.translation.x = 1.07658
        t.transform.translation.y = -1.23784
        t.transform.translation.z = 0.04381

        t.transform.rotation.x = -0.01901
        t.transform.rotation.y = 0.00215
        t.transform.rotation.z = 0.99975
        t.transform.rotation.w = -0.01119

        self.static_tf = t
        self.br.sendTransform(t)
        self.get_logger().info('Published static TF fr3_link0 -> optitrack')

        # optitrack rigid body to mujoco geom body
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'objectPushT'
        t.child_frame_id = 'objectPushT_MuJoCo'

        t.transform.translation.x = 0.0
        t.transform.translation.y = +0.025
        t.transform.translation.z = -0.0251

        quat = quaternion_from_euler(0.0, 0.0, np.pi)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)
        self.get_logger().info('Published static TF objectPushT -> objectPushT_MuJoCo')
    


    def make_jitted_step(self, ctrl):

        def step(mjx_data, policy_params,
                lin_t, quat_t,
                robot_q, robot_dq):
            """
            mjx_data, policy_params: persistent state (kept on device)
            lin_t: (3,) position (np or jnp)
            quat_t: (4,) quaternion (x,y,z,w)
            robot_q, robot_dq: (7,) joint pos/vel
            """

            # (CPU->GPU copy)
            lin_t = jnp.asarray(lin_t, dtype=jnp.float32)
            quat_t = jnp.asarray(quat_t, dtype=jnp.float32)
            robot_q = jnp.asarray(robot_q, dtype=jnp.float32)
            robot_dq = jnp.asarray(robot_dq, dtype=jnp.float32)

            # --- on device ---
            def yaw_from_quat(x, y, z, w):
                # standard ZYX Euler convention
                yaw = jnp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                return yaw + jnp.pi/2
            
            # 0 - 3 is joint
            # 7 - 13 is robot joints
                
            new_qpos = mjx_data.qpos.at[:-2].set(jnp.array([
                -lin_t[1],
                lin_t[0] -0.4,
                yaw_from_quat(x=quat_t[0], y=quat_t[1], z=quat_t[2], w=quat_t[3]) ,
                robot_q[0],
                robot_q[1],
                robot_q[2],
                robot_q[3],
                robot_q[4],
                robot_q[5],
                robot_q[6],
            ], dtype=jnp.float32))
            # zero velocit
            new_qvel = mjx_data.qvel.at[:].set(jnp.zeros(mjx_data.qvel[:].shape, dtype=jnp.float32))
            mjx_data = mjx_data.replace(
                qpos=new_qpos,
                qvel=new_qvel,
                # mocap_pos=new_mocap_pos,
                # mocap_quat=new_mocap_quat,
            )

            # resolve upadte discrepancy
            # mjx_data = jax.lax.fori_loop(
            #     0, 
            #     3, 
            #     lambda i, d: mjx.step(self.ctrl.task.model, d), 
            #     mjx_data
            # )

            # ! free object has only 6 DoF, axis angle instead of quaternion
            new_qvel = mjx_data.qvel.at[3:-2].set(jnp.array([
                robot_dq[0],
                robot_dq[1],
                robot_dq[2],
                robot_dq[3],
                robot_dq[4],
                robot_dq[5],
                robot_dq[6],
            ], dtype=jnp.float32))

            mjx_data = mjx_data.replace(    
                qvel=new_qvel,
            )
            planning_data = mjx_data

            # --- on device ---
            new_policy_params, rollouts = ctrl.optimize(planning_data, policy_params)

            return mjx_data, new_policy_params, rollouts

        return jax.jit(step, donate_argnums=(1,))


    def _update_T(self):
        """
        Update the transformation between the robot base frame and the object frame.
        
        Retrieves the current transformation from the TF buffer between "fr3_link0" 
        (robot base frame) and "objectPushT_MuJoCo" (object frame). Extracts the 
        translation and rotation components and returns them as separate numpy arrays.
        
        Returns:
            tuple: A tuple containing:
                - lin (np.ndarray): Translation vector [x, y, z] in meters as float32 array,
                                   or None if transform is not available.
                - quat (np.ndarray): Rotation quaternion [x, y, z, w] as float32 array,
                                    or None if transform is not available.
        
        Logs a warning if the TF transform is not available and returns (None, None).
        """
        if not self.tf_buffer.can_transform("fr3_link0", "objectPushT_MuJoCo",
                                            rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=0.0)):
            self.get_logger().warn("TF transform not available yet.")
            return None, None
        else:
            world_T_objReal = self.tf_buffer.lookup_transform(
                "fr3_link0", "objectPushT_MuJoCo", rclpy.time.Time()
            )
            lin = np.array([
                world_T_objReal.transform.translation.x,
                world_T_objReal.transform.translation.y,
                world_T_objReal.transform.translation.z
            ], dtype=np.float32)

            quat = np.array([
                world_T_objReal.transform.rotation.x,
                world_T_objReal.transform.rotation.y,
                world_T_objReal.transform.rotation.z,
                world_T_objReal.transform.rotation.w
            ], dtype=np.float32)

        return lin, quat


    def _get_robot_state_np(self):
        """
        Retrieve the current robot joint state as NumPy float32 arrays.
        
        Returns:
            tuple: A tuple containing:
                - robot_q (np.ndarray): Joint positions as a float32 NumPy array.
                - robot_dq (np.ndarray): Joint velocities as a float32 NumPy array.
        """
        robot_q = np.array(
            copy.deepcopy(self._current_joint_state.position),
            dtype=np.float32
        )
        robot_dq = np.array(
            copy.deepcopy(self._current_joint_state.velocity),
            dtype=np.float32
        )
        return robot_q, robot_dq
    
    
    def _run_controller(self):
        """
        Execute a single SMPC planning step.
        This method performs the following operations in sequence:
        1. Updates the T linear position and quaternion orientation
        2. Retrieves the current robot joint positions and velocities
        3. Validates that both object and robot states are available
        4. Executes the JIT-compiled policy step to update MuJoCo state and policy parameters
        5. Computes the control action from the updated policy parameters
      
        Returns:
            None
        """
        t0 = time.time()
        self.lin_t, self.quat_t = self._update_T()  
        self.robot_q, self.robot_dq = self._get_robot_state_np()
        if self.lin_t is None or self.quat_t is None:
            self.get_logger().warn(
                "Skipping control step: missing object state."
            )
            return
        if self.robot_dq is None or self.robot_q is None:
            self.get_logger().warn(
                "Skipping control step: missing robot state."
            )
            return
        
        # resolve state 
        # self.mjx_data = mjx.step(self.ctrl.task.model, self.mjx_data)

        # pose_T = np.array([self.lin_t[0], self.lin_t[1], self.lin_t[2],
        #                    self.quat_t[3], self.quat_t[0], self.quat_t[1], self.quat_t[2]])
        # ee_quat = self.get_ee_orientation()
        # ee_lin = self.get_ee_position()

        # pose_ee = np.array([ee_lin[0], ee_lin[1], ee_lin[2],
        #                    ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])
        # error = se3_left_invariant_metric(pose_T, pose_ee)
        # error = np.linalg.norm(self.lin_t - ee_lin)
        # self.get_logger().info(f"Pose error: {error}")
        
        t1 = time.time()
        self.mjx_data, self.policy_params, rollouts = self.jit_step(
            self.mjx_data,
            self.policy_params,
            self.lin_t,
            self.quat_t,
            self.robot_q,
            self.robot_dq,
        )
        self.action = self.ctrl.get_action(self.policy_params, 0.0)   
        # self.actions = np.array(self.policy_params.spline) 
        t2 = time.time()


        self.debug_data.qpos[[0, 1, 2]] = np.array([-self.lin_t[1], 
                                              self.lin_t[0] - 0.4, 
                                              np_yaw_from_quat(x=self.quat_t[0], y=self.quat_t[1], z=self.quat_t[2], w=self.quat_t[3])
                                              ])
        self.debug_data.qpos[3:-2] = self.robot_q
        self.debug_data.qvel[3:-2] = self.robot_dq
        for _ in range(3):
             mujoco.mj_step(self.debug_model, self.debug_data)

        ii = 0
        colors = np.array([
            [0.25, 0.0, 0.0, 0.4],
            [0.0, 0.25, 0.0, 0.4],
            [0.0, 0.0, 0.25, 0.4],
        ])  
        
        for k in [0,1,2]:  # 
            for i in self.trace_idxs:
                for j in range(self.ctrl.task.planning_horizon):
                    geom =self.viewer.user_scn.geoms[ii]
                    mujoco.mjv_connector(
                        geom,
                        # self.viewer.user_scn.geoms[ii],
                        mujoco.mjtGeom.mjGEOM_LINE,
                        0.5,
                        rollouts.trace_sites[0, i, j, k, :3],        # ! 
                        rollouts.trace_sites[0, i, j + 1, k, :3],    # !
                    )
                    geom.rgba[:] = colors[k, :]
                    ii += 1

        self.viewer.sync()
        t3 = time.time()
        
        self.get_logger().info(
            f"Controller step time: {t2 - t1:.3f} s"
            f" (State update: {t1 - t0:.3f} s)"
            f" (Visualization: {t3 - t2:.3f} s)"
        )
        self.last_planning_time = self.get_clock().now().nanoseconds / 1e9
    

    def _send_command(self):
        """
        Send velocity command (twist) to moveit servo
        
        :param self: Description
        """
        if self.actions is None:
            self.get_logger().warn(
                "No action available to send."
            )
            return
        idx = np.floor((self.get_clock().now().nanoseconds / 1e9 - self.last_planning_time) / self.ctrl.task.dt)
        action = self.action #self.actions[int(idx)]  # (vx, vy)
        self.get_logger().warn(
                f"Action: {action}"
            )
        pose_T = np.array([self.lin_t[0], self.lin_t[1], self.lin_t[2],
                           self.quat_t[3], self.quat_t[0], self.quat_t[1], self.quat_t[2]])
      
        # self.get_logger().info(f"Current pose_T: {pose_T}")
        pose_goal = np.array([0.45, 0.0, 0.032,
                              1.0, 0.0, 0.0, -1.0])  
        error = se3_left_invariant_metric(pose_T, pose_goal, rot_weight=1.0, trans_weight=10.0)
        self.get_logger().info(f"Pose error: {error}")
        
        if self.finished_task:
            if error > 1.0:
                self.finished_task = False
                self.get_logger().info("Resuming task...")
        else:
            if error < 0.5:
                self.finished_task = True
                self.get_logger().info("Task finished!")

        if self.finished_task:
            vx = 0.0
            vy = 0.0
            self.get_logger().warn(
                f"Finished task."
            )
        else:
            vx = 1.0 * float(action[0])
            vy = 1.0 * float(action[1])
            self.get_logger().warn(
                f"Action: {self.action}"
            )
        self.servo(linear=(vx, vy, 0.0), angular=(0.0, 0.0, 0.0))
        #self.servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))

    def _send_keyboard_command(self):
        """
        Send velocity command (twist) to moveit servo based on keyboard input.
        
        :param self: Description
        """
        if not self.teleop_enabled:
            return

        with self._key_lock:
            vx = self._key_vx
            vy = self._key_vy

        # self.get_logger().info(
        #     f"Keyboard command: vx={vx}, vy={vy}"
        # )
        self.servo(linear=(vx, vy, 0.0), angular=(0.0, 0.0, 0.0))
    
    def _states_received(self):
        """
        Check if all required states have been received at least once
        """
        if self._current_joint_state is None:
            self.get_logger().warn(
                "Current joint state not received yet."
            )
            return False
        else:
            self.get_logger().info("Current joint state received.")
            return True
    
    

if __name__ == '__main__':
    
    rclpy.init()
    max_speed = 0.3
    task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=11,
                    sim_steps_per_control_step=2,
                    ctrl_limits={"u_min": jnp.array([-max_speed, -max_speed]), 
                                 "u_max": jnp.array([max_speed, max_speed])},
                    actuation_type='velocity',
                    sampling_space="velocity",
                    block_type = 'spheres',
                )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of the walker task."
    )
    subparsers = parser.add_subparsers(
        dest="algorithm", help="Sampling algorithm (choose one)"
    )
    subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
    subparsers.add_parser("mtp", help="MTP")
    args = parser.parse_args()

    seed = 42
    num_samples = 256

    # Set the controller based on command-line arguments
    if args.algorithm is None: 
        args.algorithm = "mtp"  # Default to MTP
    elif args.algorithm == "mppi":
        print("Running MPPI")
        ctrl = MPPI(
            task,
            num_samples=num_samples,
            noise_level=0.3,
            temperature=0.6,
            num_randomizations=2,
            seed=seed,
        )
    elif args.algorithm == "mtp":
        print("Running MTP")
        ctrl = MTP(
                task,
                num_samples=num_samples,
                M=3, # horizon via control points
                N=64, # samples 
                sigma_min=0.15,
                sigma_max=0.55,
                num_elites=12,
                sigma_start=0.3,
                beta=0.2,
                alpha=0.1,
                temperature=0.1,
                interpolation='bspline',
                num_randomizations=1,
                seed=seed,
                savgol_filter=True,
                shift=True,
                planning_freq=5,
                keep_elites=1,
                default_zero_controls=True,
                update_cov=False,
            )
    
    model = task.mj_model
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as v:
      
        num_traces = 20
        trace_idxs = [i * num_traces for i in range( num_samples // num_traces)]  
        print("Indexes of traces to visualize:", trace_idxs)
        # first 50 samples
        # trace_idxs = [i for i in range(num_traces)]  # visualize only 10 traces
        # last 50 samples
        # trace_idxs = [i for i in range(num_samples - num_traces, num_samples)]  # visualize only 10 traces
        
        num_trace_sites = 3
        for i in range(
            num_trace_sites * len(trace_idxs) * ctrl.task.planning_horizon
        ):
            mujoco.mjv_initGeom(
                v.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.6, 0.6, 0.6, 0.3], dtype=np.float32),
            )
            v.user_scn.ngeom += 1

        controller = FR3_PushT(
            ctrl=ctrl,
            robot_ip="10.90.90.77",
            seed=seed,
            debug_model=model,
            debug_data=data,
            viewer=v,
            trace_idxs=trace_idxs,
        )


        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(controller)
        
        try:
            # threading.Thread(target=controller.control_loop, args=(10,), daemon=True).start()
            executor.spin()
        except KeyboardInterrupt:
            print("Shutting down controller...")
        finally:
            executor.shutdown()
            controller.destroy_node()
            rclpy.shutdown()
