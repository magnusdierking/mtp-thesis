import argparse
import copy
import pickle
import signal
import threading
import time

import jax
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R

from hydrax.algs import CEM, MPPI, MTP, PredictiveSampling
from hydrax.risk import (
    AverageCost,
    ConditionalValueAtRisk,
    ExpectedCost,
    InverseConditionalValueAtRisk,
    InverseValueAtRisk,
    ValueAtRisk,
)
from hydrax.tasks.pusht_franka import PushTFranka
from hydrax.utils.files import get_data_path
from hydrax.utils.utils import mat2quat, quat_error_body, se3_left_invariant_metric


jax.config.update("jax_platform_name", "gpu")
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from franka_panda_server import FrankaPandaServer
from geometry_msgs.msg import TransformStamped
from mujoco import mjx
from pynput import keyboard
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from shape_msgs.msg import SolidPrimitive
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler

from hydrax.utils.files import get_data_path, get_root_path


def np_yaw_from_quat(x, y, z, w):
    # standard ZYX Euler convention
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw + np.pi / 2


class FR3_PushT(FrankaPandaServer):
    def __init__(
        self,
        ctrl,
        robot_ip,
        seed,
        debug_model,
        debug_data,
        viewer,
        trace_idxs,
        planning_freq=10,
        run_time_sec=40.0,
    ):

        super().__init__(robot_ip, None)

        self.get_logger().info("Initializing SMPC Controller...")
        self.ctrl = ctrl
        self.seed = seed
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
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        self.add_collision_primitive(
            id="wall x",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(0.05, 1.0, 0.5),
            position=np.array([1.025, 0.0, 0.15]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        self.add_collision_primitive(
            id="wall y_neg",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(1.2, 0.05, 0.5),
            position=np.array([0.4, -0.475, 0.15]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        self.add_collision_primitive(
            id="wall y_pos",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(1.2, 0.05, 0.5),
            position=np.array([0.4, 0.475, 0.15]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )

        ####################################
        ##       Move to initial pose     ##
        ####################################
        # 10-14 seeds
        self.init_pos = np.array(
            [
                0.55 + np.random.uniform(-0.03, 0.03),
                -0.05 + np.random.uniform(-0.03, 0.03),
                0.045,
            ]
        )
        # 20-24 seeds
        # self.init_pos = np.array([0.5 + np.random.uniform(-0.03 , 0.03),
        #                           0.0 + np.random.uniform(-0.03, 0.03),
        #                           0.045])
        # 30-34 seeds
        # self.init_pos = np.array([0.5 + np.random.uniform(-0.03 , 0.03),
        #                           0.0 + np.random.uniform(-0.03, 0.03),
        #                           0.045])

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

        self.tf_buffer = self._tf_buffer
        self.tf_listener = self._tf_listener

        # Static TF broadcaster
        self.br = StaticTransformBroadcaster(self)
        self._publish_static_robot_tf()

        # Wait until TF is available
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tf_buffer.can_transform(
                "fr3_link0",
                "objectPushT_MuJoCo",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.0),
            ):
                break
        time.sleep(0.5)  # wait a bit more

        ####################################
        ##         JIT Controller         ##
        ####################################

        # Wait until all states are received
        while not self._states_received():
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("All states received, initializing controller...")
        self.trace_idxs = trace_idxs

        id = mujoco.mj_name2id(
            self.ctrl.task.mj_model, mujoco.mjtObj.mjOBJ_BODY, "ghost_block"
        )
        self.ghost_id = self.ctrl.task.mj_model.body_mocapid[id]

        #! mj_model on CPU
        #! model on GPu
        # Create mjx_data on host,  push to device
        
        self.mjx_data = mjx.make_data(self.ctrl.model)
        self.policy_params = self.ctrl.init_params(seed)

        self.get_logger().info(
            f"Planning with {self.ctrl.task.planning_horizon} steps "
            f"over a {self.ctrl.task.planning_horizon * self.ctrl.task.dt} second horizon."
        )

        while (
            self.lin_t is None
            or self.quat_t is None
            or self.robot_q is None
            or self.robot_dq is None
        ):
            rclpy.spin_once(self, timeout_sec=0.1)
            self.lin_t, self.quat_t = self._update_T()
            self.robot_q, self.robot_dq = self._get_robot_state_np()
        self.get_logger().info(
            "Initial object and robot states received, starting controller..."
        )

        # Do a forward once (host side is fine here)
        self.mjx_data = mjx.forward(self.ctrl.model, self.mjx_data)
        
        # add domain randomization
        self._domain_randomize_mjx_model()
        self.dt = self.debug_model.opt.timestep

        # Make unified jitted step function
        self.jit_step = self.make_jitted_step(ctrl)

        self.get_logger().info("Jitting controller...")
        st = time.time()

        self.debug_data.qpos[0:7] = np.array(
            [
                self.lin_t[0],
                self.lin_t[1],
                self.lin_t[2],
                self.quat_t[3],
                self.quat_t[0],
                self.quat_t[1],
                self.quat_t[2],
            ]
        )
        self.debug_data.qpos[7:14] = self.robot_q
        self.debug_data.qvel[6:13] = self.robot_dq
        self.debug_data.time = self.get_clock().now().nanoseconds / 1e9

        # for early visualization
        # mujoco.mj_forward(self.debug_model, self.debug_data)
        mujoco.mj_step(self.debug_model, self.debug_data)
        self.viewer.sync()

        # One call to transfer mjx_data & policy_params to GPU and compile
        self.jit_step = self.jit_step.lower(
            self.mjx_data,
            self.policy_params,
            self.debug_data.qpos,
            self.debug_data.qvel,
            self.debug_data.mocap_pos,
            self.debug_data.mocap_quat,
            self.get_clock().now().nanoseconds / 1e9,
        ).compile()

        # warmstart simulation
        # for _ in range(5):
        #     self.mjx_data = jax.vmap(
        #         mjx.forward,
        #         in_axes=(self.ctrl.randomized_axes, None)
        #     )(self.ctrl.model, self.mjx_data)
            # self.mjx_data = mjx.step(self.ctrl.model, self.mjx_data)
        # warmstart controller
        for _ in range(1):
            self.mjx_data, self.policy_params, _ = self.jit_step(
                self.mjx_data,
                self.policy_params,
                self.debug_data.qpos,
                self.debug_data.qvel,
                self.debug_data.mocap_pos,
                self.debug_data.mocap_quat,
                self.get_clock().now().nanoseconds / 1e9,
            )

        self.get_logger().info(f"Time to jit and warmstart: {time.time() - st:.3f} s")

        ####################################
        ##    Start keyboard Controller   ##
        ####################################
        self._key_lock = threading.Lock()
        self.teleop_enabled = False
        self._key_vx = 0.0
        self._key_vy = 0.0
        self._key_entered = False

        # Start keyboard listener in background
        threading.Thread(target=self._keyboard_loop, daemon=True).start()

        ####################################
        ##           Start Timer          ##
        ####################################
        self.shutdown_flag = threading.Event()  # Thread-safe flag
        signal.signal(signal.SIGINT, self.signal_handler)
        # Logs
        self.log = []
        self.ctr = 0

        self.finished_task = False
        self.servo_freq = 30  # Hz
        self.plan_freq = planning_freq  # Hz ! needs to be lower than max (GIL)
        self.actions = None
        self.predicted_states = None
        self.domain_weights = None
        self.action = None

        self.mocap_T_bids = controller.task.get_mocap_T_bids()

        self.sim_group = MutuallyExclusiveCallbackGroup()

        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.last_planning_time = self.get_clock().now().nanoseconds / 1e9

        self.create_timer(
            1.0 / self.plan_freq, self._run_controller, callback_group=self.sim_group
        )
        time.sleep(0.2)
        self.create_timer(
            1.0 / self.servo_freq,
            self._send_keyboard_command,
            callback_group=self.parallel_group,
        )
        self.run_time_sec = run_time_sec
        self.create_timer(self.run_time_sec, self._timeout_callback)





    def _domain_randomize_mjx_model(self):
        top_masses = jnp.linspace(0.01, 0.35, self.ctrl.num_randomizations)
        bottom_masses = jnp.linspace(0.1, 1.0, self.ctrl.num_randomizations)

        self.randomization_matrix = jnp.array(
            [
                top_masses,
                bottom_masses,
            ]
        )

        body_id = mujoco.mj_name2id(self.debug_model, mujoco.mjtObj.mjOBJ_BODY, "block")

        randomized_axes = [
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_invweight0",
            "dof_invweight0",
            "dof_M0",
            # "dof_armature",
            "body_subtreemass",
        ]

        derived_values = {field: [] for field in randomized_axes}

        for top_mass, bottom_mass in zip(top_masses, bottom_masses):
            spec = mujoco.MjSpec.from_file(
                (
                    get_root_path()
                    / "hydrax"
                    / "models"
                    / "fr3_pushT_vel"
                    / "scene_mjx_free_dr.xml"
                ).as_posix()
            )
            body = spec.body("block")

            # Set mass for each geom by name
            for geom in body.geoms:
                if geom.name == "top":
                    geom.mass = float(top_mass)
                    print(f"Set 'top' geom mass: {top_mass}")
                elif geom.name == "bottom":
                    geom.mass = float(bottom_mass)
                    print(f"Set 'bottom' geom mass: {bottom_mass}")

            compiled_model = spec.compile()
            total_mass = compiled_model.body_mass[body_id]
            print(
                f"Compiled body mass (top={top_mass:.3f} + bottom={bottom_mass:.3f}): {total_mass:.3f}"
            )
            print(f"Compiled inertia: {compiled_model.body_inertia[body_id]}")
            print(f"Compiled ipos: {compiled_model.body_ipos[body_id]}\n")

            # Append derived quantities to lists
            derived_values["body_mass"].append(compiled_model.body_mass)
            derived_values["body_inertia"].append(compiled_model.body_inertia)
            derived_values["body_ipos"].append(compiled_model.body_ipos)
            derived_values["body_invweight0"].append(compiled_model.body_invweight0)
            derived_values["dof_invweight0"].append(compiled_model.dof_invweight0)
            derived_values["dof_M0"].append(compiled_model.dof_M0)
            # derived_values["dof_armature"].append(compiled_model.dof_armature)
            derived_values["body_subtreemass"].append(compiled_model.body_subtreemass)

        # Turn all lists in derived_values into arrays
        for field in derived_values:
            derived_values[field] = jnp.array(derived_values[field])

        print("\n=== Final Randomized Values ===")
        for field, values in derived_values.items():
            print(f"{field} (block body): {values[:, body_id]}")

        self.ctrl.update_randomized_axes(randomized_axes)

        self.ctrl.model = self.ctrl.model.replace(**derived_values)

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
                if key.char == "t":
                    self.teleop_enabled = not self.teleop_enabled
                    self.get_logger().info(f"teleop_enabled = {self.teleop_enabled}")
                    return
                if key.char == " ":
                    # space = stop
                    with self._key_lock:
                        self._key_vx = 0.0
                        self._key_vy = 0.0
                    return
            except AttributeError:
                pass

            vx, vy = 0.0, 0.0
            key_entered = False

            if key == keyboard.Key.up:
                vx = 0.1
            elif key == keyboard.Key.down:
                vx = -0.1
            elif key == keyboard.Key.left:
                vy = +0.1
            elif key == keyboard.Key.right:
                vy = -0.1
            elif key == keyboard.Key.space:
                key_entered = True
            else:
                return
            with self._key_lock:
                self._key_entered = key_entered
                self._key_vx = vx
                self._key_vy = vy

        def on_release(key):
            # When arrow key released -> stop (simple behavior)
            if key in (
                keyboard.Key.up,
                keyboard.Key.down,
                keyboard.Key.left,
                keyboard.Key.right,
            ):
                with self._key_lock:
                    self._key_vx = 0.0
                    self._key_vy = 0.0
            if key == keyboard.Key.space:
                with self._key_lock:
                    self._key_entered = False

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def _timeout_callback(self):
        self.get_logger().info("Timeout reached, shutting down...")
        # Any controller-specific cleanup, if needed
        self.send_stop_command()
        rclpy.shutdown()

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
        t.header.frame_id = "fr3_link0"
        t.child_frame_id = "optitrack"

        t.transform.translation.x = 1.11564  #1.07658 
        t.transform.translation.y = -1.26901 #-1.23784
        t.transform.translation.z = -0.0264  # 0.04381

        t.transform.rotation.x = -0.01411     #-0.01901
        t.transform.rotation.y = -0.00319     # 0.00215
        t.transform.rotation.z = 0.99984     # 0.99975
        t.transform.rotation.w = 0.01083     # -0.01119

        self.static_tf = t
        self.br.sendTransform(t)
        self.get_logger().info("Published static TF fr3_link0 -> optitrack")

        # optitrack rigid body to mujoco geom body
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "objectPushT"
        t.child_frame_id = "objectPushT_MuJoCo"

        t.transform.translation.x = 0.02075
        t.transform.translation.y = -0.011
        t.transform.translation.z = -0.0285

        # quat = quaternion_from_euler(-0.05, 0.02, np.pi)
        quat = quaternion_from_euler(0.0, -0.02, -np.pi)

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)
        self.get_logger().info("Published static TF objectPushT -> objectPushT_MuJoCo")

    def make_jitted_step(self, ctrl):

        def step(mjx_data, policy_params, q, dq, mocap_pos, mocap_quat, curr_time):

            mjx_data = mjx_data.replace(
                qpos=jnp.array(q),
                qvel=jnp.array(dq),
                mocap_pos=jnp.array(mocap_pos),
                mocap_quat=jnp.array(mocap_quat),
                time=curr_time,
            )
            # vmapped_forward = jax.vmap(
            #     mjx.forward,
            #     in_axes=(self.ctrl.randomized_axes, None)
            # )
            # mjx_data = vmapped_forward(self.ctrl.model, mjx_data)
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
        if not self.tf_buffer.can_transform(
            "fr3_link0",
            "objectPushT_MuJoCo",
            rclpy.time.Time(),
            rclpy.duration.Duration(seconds=0.0),
        ):
            self.get_logger().warn("TF transform not available yet.")
            return None, None
        world_T_objReal = self.tf_buffer.lookup_transform(
            "fr3_link0", "objectPushT_MuJoCo", rclpy.time.Time()
        )
        lin = np.array(
            [
                world_T_objReal.transform.translation.x,
                world_T_objReal.transform.translation.y,
                world_T_objReal.transform.translation.z,
            ],
            dtype=np.float32,
        )

        quat = np.array(
            [
                world_T_objReal.transform.rotation.x,
                world_T_objReal.transform.rotation.y,
                world_T_objReal.transform.rotation.z,
                world_T_objReal.transform.rotation.w,
            ],
            dtype=np.float32,
        )

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
            copy.deepcopy(self._current_joint_state.position), dtype=np.float32
        )
        robot_dq = np.array(
            copy.deepcopy(self._current_joint_state.velocity), dtype=np.float32
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
            self.get_logger().warn("Skipping control step: missing object state.")
            return
        if self.robot_dq is None or self.robot_q is None:
            self.get_logger().warn("Skipping control step: missing robot state.")
            return
        current_time = self.get_clock().now().nanoseconds / 1e9
        sim_z = self.debug_data.qpos[2]  # current object height in simulation
        # resolve state
        self.debug_data.qpos[0:7] = np.array(
            [
                self.lin_t[0],
                self.lin_t[1],
                self.lin_t[2],
                self.quat_t[3],
                self.quat_t[0],
                self.quat_t[1],
                self.quat_t[2],
            ]
        )
        self.debug_data.qpos[7:14] = self.robot_q
        self.debug_data.qvel[6:13] = self.robot_dq
        self.debug_data.time = current_time

        # Domain Randomization - compare actual observation to predictions
        if self.predicted_states is not None:
            # observation of T state over domains
            ref_site = self.predicted_states[
                :, -1, ...
            ]  # (domains, 7)
            for bid, idx in zip(self.mocap_T_bids, range(len(self.mocap_T_bids)), strict=True):
                self.debug_data.mocap_pos[bid] = ref_site[idx, :3]
                self.debug_data.mocap_quat[bid] = ref_site[idx, 3:]
                # print(f"Setting mocap bid {bid} to {ref_site[idx, :3]}, {ref_site[idx, 3:]}")
            # rest of mocap copies T
            if len(self.mocap_T_bids) < 24:
                bid = mujoco.mj_name2id(self.debug_model, mujoco.mjtObj.mjOBJ_BODY, "block")
                for bid in range(len(self.mocap_T_bids), 24):
                    self.debug_data.mocap_pos[bid] = self.debug_data.xpos[bid]
                    self.debug_data.mocap_quat[bid] = self.debug_data.xquat[bid]
            
            # compute error to actual observtion
            current_obs = np.array(self.debug_data.qpos[:7], dtype=np.float32)
            distances = jax.vmap(se3_left_invariant_metric, in_axes=(0, None))(
                        self.predicted_states[..., 4, :], current_obs
                    )


        # update sites etc.
        # mujoco.mj_forward(self.debug_model, self.debug_data)
        mujoco.mj_step(self.debug_model, self.debug_data)

        t1 = time.time()
        self.mjx_data, self.policy_params, rollouts = self.jit_step(
            self.mjx_data,
            self.policy_params,
            self.debug_data.qpos,
            self.debug_data.qvel,
            self.debug_data.mocap_pos,
            self.debug_data.mocap_quat,
            current_time,
        )
        # self.action = self.ctrl.get_action(self.policy_params, 0.0)
        self.actions = np.array(self.policy_params.spline)
        self.predicted_states = np.array(self.policy_params.predicted_state)
        self.domain_weights = np.array(self.policy_params.domain_weights)
        self.action = self.actions[0]
        t2 = time.time()

        terminal_error = self.ctrl.task.running_cost(self.debug_data)
        ee_position = self.ctrl.task._get_ee_position(self.debug_data)

        self.last_planning_time = self.get_clock().now().nanoseconds / 1e9
        self.ctr += 1

        self.get_logger().info(f"Step {self.ctr}: Domains error: {distances:.4f}")

        self.log.append(
            {
                "time": self.last_planning_time - self.start_time,
                "planning_time": t2 - t1,
                "object_pos": self.lin_t.tolist(),
                "object_quat": self.quat_t.tolist(),
                "ee_pos": ee_position.tolist(),
                "terminal_error": float(terminal_error),
                "action": self.action.tolist(),
            }
        )
        ii = 0
        # colors = np.array([
        #     [0.25, 0.0, 0.0, 0.4],
        #     [0.0, 0.25, 0.0, 0.4],
        #     [0.0, 0.0, 0.25, 0.4],
        # ])
        for k in [2]:  #
            for i in self.trace_idxs:
                for j in range(self.ctrl.task.planning_horizon):
                    geom = self.viewer.user_scn.geoms[ii]
                    mujoco.mjv_connector(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        0.5,
                        rollouts.trace_sites[0, i, j, k, :3],  # !
                        rollouts.trace_sites[0, i, j + 1, k, :3],  # !
                    )
                    # geom.rgba[:] = colors[k, :]
                    ii += 1
        self.viewer.sync()

        t3 = time.time()

        self.get_logger().info(
            f"Controller step time: {t2 - t1:.3f} s"
            f" (State update: {t1 - t0:.3f} s)"
            f" (Visualization: {t3 - t2:.3f} s)"
            f" Error: {terminal_error:.4f}"
        )

    def _send_keyboard_command(self):
        """
        Send velocity command (twist) to moveit servo based on keyboard input.

        :param self: Description
        """
        if self.shutdown_flag.is_set():
            self._timeout_callback()
            return
        if not self.teleop_enabled:
            vx = 1.0 * float(self.action[0])
            vy = 1.0 * float(self.action[1])
            self.get_logger().info(f"SMPC action: vx={vx:.3f}, vy={vy:.3f}")
            # self.servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
            self.servo(linear=(vx, vy, 0.0), angular=(0.0, 0.0, 0.0))

        else:
            with self._key_lock:
                vx = self._key_vx
                vy = self._key_vy
                key_entered = self._key_entered

            if self._key_entered:
                vx = 0.0
                vy = 0.0
            self.servo(linear=(vx, vy, 0.0), angular=(0.0, 0.0, 0.0))

    def _states_received(self):
        """
        Check if all required states have been received at least once
        """
        if self._current_joint_state is None:
            self.get_logger().warn("Current joint state not received yet.")
            return False
        self.get_logger().info("Current joint state received.")
        return True

    def save_log(self, path, filename="fr3_pusht_"):
        log_file = path / f"{filename}{self.ctrl.__class__.__name__}_seed{self.seed}"

        with open(str(log_file) + ".pkl", "wb") as f:
            pickle.dump(self.log, f)

    def send_stop_command(self):
        """
        Send a stop command to the robot to halt any ongoing motion.
        """
        self.get_logger().info("Sent stop command to the robot.")
        self.servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))

    def signal_handler(self, sig, frame):
        self.get_logger().info("SIGINT received")
        self.shutdown_flag.set()


if __name__ == "__main__":
    rclpy.init()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of the walker task."
    )
    subparsers = parser.add_subparsers(
        dest="algorithm", help="Sampling algorithm (choose one)"
    )
    subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
    subparsers.add_parser("mtp", help="MTP")
    subparsers.add_parser("cem", help="Cross-Entropy Method")
    subparsers.add_parser("ps", help="Predictive Sampling")
    args = parser.parse_args()

    seed = 5555
    # PS 10

    num_samples = 200  # 512
    NUM_RANDOMIZATIONS = 10
    planning_freq = 5  # Hz

    parser.add_argument(
        "--risk",
        choices=["average", "expectation", "var", "cvar", "ivar", "icvar"],
        help="risk aggregation method for domain randomization",
    )

    args = parser.parse_args()
    print(args)

    risk_alpha = 0.5  # for (C)VaR
    uniform_weights = (
        jnp.ones((NUM_RANDOMIZATIONS,), dtype=jnp.float32) / NUM_RANDOMIZATIONS
    )
    if args.risk is None or args.risk == "average":
        args.risk = "average"
        aggregation = AverageCost()
    elif args.risk == "expectation":
        # initialize with uniform weights
        aggregation = ExpectedCost(weights=uniform_weights)
    elif args.risk == "var":
        aggregation = ValueAtRisk(alpha=risk_alpha)
    elif args.risk == "cvar":
        aggregation = ConditionalValueAtRisk(alpha=risk_alpha, weights=uniform_weights)
    elif args.risk == "ivar":
        aggregation = InverseValueAtRisk(alpha=risk_alpha)
    elif args.risk == "icvar":
        aggregation = InverseConditionalValueAtRisk(
            alpha=risk_alpha, weights=uniform_weights
        )

    max_speed = 0.3
    task = PushTFranka(
        ik_type="pinv",
        planning_horizon=10,
        sim_steps_per_control_step=2,
        ctrl_limits={
            "u_min": jnp.array([-max_speed, -max_speed]),
            "u_max": jnp.array([max_speed, max_speed]),
        },
        actuation_type="velocity",
        sampling_space="velocity",
        block_type="dr-free",
    )

    # Set the controller based on command-line arguments
    if args.algorithm is None:
        args.algorithm = "mtp"  # Default to MTP
    elif args.algorithm == "mppi":
        print("Running MPPI")
        ctrl = MPPI(
            task,
            num_samples=num_samples,
            alpha=0.1,
            temperature=0.1,
            noise_level=0.3,
            num_randomizations=NUM_RANDOMIZATIONS,
            savgol_filter=True,
            shift=True,
            planning_freq=planning_freq,
            seed=seed,
            update_cov=False,
            risk_strategy=aggregation,
        )
    elif args.algorithm == "cem":
        print("Running CEM")
        ctrl = CEM(
            task,
            alpha=0.1,
            num_samples=num_samples,
            sigma_start=0.3,
            sigma_min=0.05,
            num_elites=72,
            num_randomizations=NUM_RANDOMIZATIONS,
            savgol_filter=True,
            shift=True,
            planning_freq=planning_freq,
            seed=seed,
            update_cov=False,
            risk_strategy=aggregation,
        )
    elif args.algorithm == "ps":
        print("Running Predictive Sampling")
        ctrl = PredictiveSampling(
            task,
            num_samples=num_samples,
            num_randomizations=NUM_RANDOMIZATIONS,
            noise_level=0.3,
            savgol_filter=True,
            shift=True,
            planning_freq=planning_freq,
            seed=seed,
            risk_strategy=aggregation,
        )
    elif args.algorithm == "mtp":
        print("Running MTP")
        ctrl = MTP(
            task,
            num_samples=num_samples,
            temperature=0.1,
            M=3,  # horizon via control points
            N=64,  # samples
            sigma_min=0.15,
            sigma_max=0.55,
            num_elites=72,
            sigma_start=0.3,
            beta=0.25,
            alpha=0.1,
            interpolation="bspline",
            num_randomizations=NUM_RANDOMIZATIONS,
            seed=seed,
            savgol_filter=True,
            shift=True,
            planning_freq=planning_freq,
            keep_elites=1,
            default_zero_controls=False,
            update_cov=False,
            risk_strategy=aggregation,
        )

    model = task.mj_model # CPU model
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as v:
        num_traces = 0
        trace_idxs = np.linspace(0, num_samples - 1, num=num_traces, dtype=int).tolist()
        # trace_idxs.extend( [i * (num_samples // (num_traces -1)) for i in range(1, num_traces -1)] )
        # trace_idxs = [i * num_traces for i in range( num_samples // num_traces)]
        print("Indexes of traces to visualize:", trace_idxs)
        # first 50 samples
        # trace_idxs = [i for i in range(num_traces)]  # visualize only 10 traces
        # last 50 samples
        # trace_idxs = [i for i in range(num_samples - num_traces, num_samples)]  # visualize only 10 traces

        num_trace_sites = 1
        for i in range(num_trace_sites * len(trace_idxs) * ctrl.task.planning_horizon):
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
            planning_freq=planning_freq,
            trace_idxs=trace_idxs,
        )

        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(controller)

        try:
            executor.spin()
        except KeyboardInterrupt:
            print("Shutting down controller...")
        finally:
            executor.shutdown()
            path = get_data_path() / "sim-real-free"
            # controller.save_log(path)
            controller.destroy_node()
            rclpy.shutdown()
