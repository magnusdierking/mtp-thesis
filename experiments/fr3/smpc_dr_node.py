"""
SMPC Planner Node — publishes action splines for the servo node.

This node handles:
  - MoCap TF lookups for the T-block
  - Joint state subscriptions (via FrankaPandaServer)
  - Initial robot positioning (plan_and_move_to_pose)
  - JAX/MJX domain-randomised planning on GPU
  - Debug MuJoCo viewer visualisation
  - Publishing the action spline + dt on /smpc/action_spline

It does NOT send servo commands.  A separate lightweight servo node
subscribes to the spline topic and indexes into it at high frequency.

Usage:
    python smpc_planner_node.py mtp --risk average
"""

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
from hydrax.utils.utils import se3_left_invariant_metric

jax.config.update("jax_platform_name", "gpu")
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from franka_panda_server import FrankaPandaServer
from geometry_msgs.msg import TransformStamped
from mujoco import mjx
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler

from hydrax.utils.files import get_root_path

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def compute_randomizations(
    batched_model: mjx.Model,
    mj_model: mujoco.MjModel,
    randomization_specs: dict,
) -> tuple[mjx.Model, list]:
    spec = mujoco.MjSpec()
    randomized_axes = []
    for type, type_dict in randomization_specs.items():
        
        for name, param_dict in type_dict.items():
            if type == 'joints':
                id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            elif type == 'bodies':
                id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            elif type == 'geoms':
                id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            else:
                raise ValueError(f"Unknown type '{type}' in randomization specs")
            for field_name in param_dict:
                if hasattr(batched_model, field_name):
                    local_idx, randomization_values = param_dict[field_name] 
                    field_array = getattr(batched_model, field_name)
                    nbr_randomizations = randomization_values.shape[0] 
                    
                    if field_array.shape[0] != nbr_randomizations:
                        reps = (nbr_randomizations,) + (1,) * field_array.ndim
                        field_array = jnp.tile(field_array, reps)
                    if local_idx is None:
                        field_array = field_array.at[:, [id]].set(jnp.expand_dims(randomization_values, axis=-1))
                    else:
                        field_array = field_array.at[:, id, [local_idx]].set(jnp.expand_dims(randomization_values, axis=-1))


                    # randomizations[field_name] = field_array
                    batched_model = batched_model.replace(**{field_name: field_array})
                    print(f"Setting type  '{type}' field '{field_name}' for {name} index {id}")
                    randomized_axes.append(field_name)
                else:
                    raise ValueError(f"{type.capitalize()} {name} has no attribute '{field_name}'")
    

    return batched_model, randomized_axes              
                    
                

def np_yaw_from_quat(x, y, z, w):
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw + np.pi / 2


def quat_to_angular_velocity(q_prev, q_curr, dt):
    """Finite-difference angular velocity from two quaternions ([x,y,z,w])."""
    if dt <= 0:
        return np.zeros(3, dtype=np.float32)
    r_prev = R.from_quat(q_prev)
    r_curr = R.from_quat(q_curr)
    r_delta = r_curr * r_prev.inv()
    rotvec = r_delta.as_rotvec()
    return (rotvec / dt).astype(np.float32)


# ---------------------------------------------------------------------------
#  Planner Node
# ---------------------------------------------------------------------------


class SMPCPlannerNode(FrankaPandaServer):
    """
    Planning-only node.  Publishes the action spline on ``/smpc/action_spline``
    as a ``Float32MultiArray``.

    Message layout
    --------------
    data = [dt, vx_0, vy_0, vx_1, vy_1, …, vx_{H-1}, vy_{H-1}]

    where *dt* is the time-step per spline index (seconds) and H is the
    planning horizon.  The servo node reconstructs the (H, 2) spline from
    ``data[1:]`` and uses ``data[0]`` for index timing.
    """

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
        run_time_sec=30.0,
    ):
        super().__init__(robot_ip=robot_ip, gripper_type=None)
        self.get_logger().info("Initializing SMPC Planner Node …")

        self.ctrl = ctrl
        self.seed = seed
        self.debug_model = debug_model
        self.debug_data = debug_data
        self.viewer = viewer

        # wait for node to initialize in the background
        time.sleep(1.0) 

        # ---- MoveIt safety constraints ----
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

        # ---- Move to initial pose ----
        self.init_pos = np.array(
            [
                0.5 + np.random.uniform(-0.03, 0.03),
                -0.15 + np.random.uniform(-0.03, 0.03),
                0.045,
            ]
        )
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.init_rot = R.from_quat(self.init_quat).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = self.init_rot
        pose[:3, 3] = self.init_pos
        self.plan_and_move_to_pose(pose)

        # ---- Object / robot state holders ----
        self.lin_t = None
        self.quat_t = None
        self.robot_q = None
        self.robot_dq = None

        # Previous MoCap observations for velocity estimation
        self._prev_lin_t = None
        self._prev_quat_t = None
        self._prev_obs_time = None

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
        time.sleep(0.5)

        # ---- Wait for joint states ----
        while not self._states_received():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("All states received, initialising controller …")

        self.trace_idxs = trace_idxs

        ghost_body_id = mujoco.mj_name2id(
            self.ctrl.task.mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "ghost_block",
        )
        self.ghost_id = self.ctrl.task.mj_model.body_mocapid[ghost_body_id]

        # Create mjx data
        self.mjx_data = mjx.make_data(self.ctrl.model)
        self.policy_params = self.ctrl.init_params(seed)

        self.get_logger().info(
            f"Planning with {self.ctrl.task.planning_horizon} steps "
            f"over a {self.ctrl.task.planning_horizon * self.ctrl.task.dt} s horizon."
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
        self.get_logger().info("Initial object + robot states received.")

        # Initialise velocity estimation baseline
        self._prev_lin_t = self.lin_t.copy()
        self._prev_quat_t = self.quat_t.copy()
        self._prev_obs_time = self.get_clock().now().nanoseconds / 1e9

        # Forward pass on host
        self.mjx_data = mjx.forward(self.ctrl.model, self.mjx_data)

        # Domain randomisation
        self._domain_randomize_mjx_model()
        self.dt = self.debug_model.opt.timestep

        # JIT compile
        self.jit_step = self._make_jitted_step(ctrl)
        self.get_logger().info("Jitting controller …")
        st = time.time()

        self._write_debug_state(
            self.lin_t,
            self.quat_t,
            self.robot_q,
            self.robot_dq,
            np.zeros(6, dtype=np.float32),
            self.get_clock().now().nanoseconds / 1e9,
        )
        mujoco.mj_forward(self.debug_model, self.debug_data)
        self.viewer.sync()

        self.jit_step = self.jit_step.lower(
            self.mjx_data,
            self.policy_params,
            self.debug_data.qpos,
            self.debug_data.qvel,
            self.debug_data.mocap_pos,
            self.debug_data.mocap_quat,
            self.get_clock().now().nanoseconds / 1e9,
        ).compile()

        # Warm-start
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
        self.get_logger().info(f"JIT + warm-start: {time.time() - st:.3f} s")

        # ---- Spline publisher ----
        self.spline_pub = self.create_publisher(
            Float32MultiArray,
            "/smpc/action_spline",
            10,
        )

        # ---- State ----
        self.shutdown_flag = threading.Event()
        signal.signal(signal.SIGINT, self.signal_handler)

        self.log = []
        self.ctr = 0
        self.predicted_states = None
        self.domain_weights = None
        self.mocap_T_bids = self.ctrl.task.get_mocap_T_bids()

        self.plan_freq = planning_freq
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        # ---- Launch planning thread ----
        self._planning_thread = threading.Thread(
            target=self._planning_loop,
            daemon=True,
        )
        self._planning_thread.start()

        # ---- Timeout ----
        self.run_time_sec = run_time_sec
        self.create_timer(self.run_time_sec, self._timeout_callback)

    # ------------------------------------------------------------------
    #  Domain randomisation
    # ------------------------------------------------------------------

    def _domain_randomize_mjx_model(self):

        randomization_dict = {
            "geoms": {
                # "ground": {
                #     "geom_friction": (0, jnp.linspace(0.01, 5.0, NUM_RANDOMIZATIONS)),
                # },
                # "bottom": {
                #     "geom_friction": (0, jnp.linspace(0.01, 5.0, NUM_RANDOMIZATIONS)),
                # },
                # "vertical": {
                #     "geom_friction": (0, jnp.linspace(0.01, 5.0, NUM_RANDOMIZATIONS)),
                # },
                "ee": {
                    # "geom_margin": (None, jnp.linspace(-0.015, 0.015, NUM_RANDOMIZATIONS)),
                    "geom_margin": (None, jnp.linspace(-0.0, 0.0, NUM_RANDOMIZATIONS)),

                },
            }
        }

        ctrl.model, randomized_axes = compute_randomizations(
            ctrl.model,
            self.debug_model,
            randomization_dict,
        )
        ctrl.update_randomized_axes(randomized_axes)
        print(ctrl.model.geom_margin)
        # top_masses = jnp.array([0.03, 0.06, 0.15, 0.4, 0.5, 0.8, 1.3, 2.3, 2.5, 4.0])   
        # bottom_masses = jnp.array([0.03, 0.09, 0.15, 0.5, 0.4, 1.0, 2.3, 1.3, 2.5, 6.0])
        # top_masses = jnp.linspace(0.15, 0.15, self.ctrl.num_randomizations)
        # bottom_masses = jnp.linspace(0.02, 0.8, self.ctrl.num_randomizations)


        # self.randomization_matrix = jnp.array([top_masses, bottom_masses])

        # body_id = mujoco.mj_name2id(
        #     self.debug_model,
        #     mujoco.mjtObj.mjOBJ_BODY,
        #     "block",
        # )

        # randomized_axes = [
        #     "body_mass",
        #     "body_inertia",
        #     "body_ipos",
        #     "body_invweight0",
        #     "dof_invweight0",
        #     "dof_M0",
        #     "dof_armature",
        #     "body_subtreemass",
        # ]
        # derived_values = {f: [] for f in randomized_axes}

        # xml_path = (
        #     get_root_path()
        #     / "hydrax"
        #     / "models"
        #     / "fr3_pushT_vel"
        #     / "scene_mjx_free_dr.xml"
        # ).as_posix()

        # for top_mass, bottom_mass in zip(top_masses, bottom_masses):
        #     spec = mujoco.MjSpec.from_file(xml_path)
        #     body = spec.body("block")
        #     for geom in body.geoms:
        #         if geom.name == "top":
        #             geom.mass = float(top_mass)
        #         elif geom.name == "bottom":
        #             geom.mass = float(bottom_mass)
        #     compiled = spec.compile()
        #     self.get_logger().info(
        #         f"DR: top={float(top_mass):.3f}  bottom={float(bottom_mass):.3f}  "
        #         f"total={compiled.body_mass[body_id]:.3f}"
        #     )
        #     for field in randomized_axes:
        #         derived_values[field].append(getattr(compiled, field))

        # for field in derived_values:
        #     derived_values[field] = jnp.array(derived_values[field])

        # self.ctrl.update_randomized_axes(randomized_axes)
        # self.ctrl.model = self.ctrl.model.replace(**derived_values)

    # ------------------------------------------------------------------
    #  Static transforms
    # ------------------------------------------------------------------

    def _publish_static_robot_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "fr3_link0"
        t.child_frame_id = "optitrack"
        t.transform.translation.x = 1.11670  
        t.transform.translation.y = -1.26868 
        t.transform.translation.z = -0.0252  

        t.transform.rotation.x = -0.01299     
        t.transform.rotation.y = -0.00261    
        t.transform.rotation.z = 0.99987     
        t.transform.rotation.w = 0.00923     
        self.static_tf = t
        self.br.sendTransform(t)
        self.get_logger().info("Published static TF fr3_link0 → optitrack")

        t2 = TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = "objectPushT"
        t2.child_frame_id = "objectPushT_MuJoCo"
        t2.transform.translation.x = 0.0255
        t2.transform.translation.y = 0.00
        t2.transform.translation.z = -0.034
        quat = quaternion_from_euler(0.0, 0.0, -np.pi)
        t2.transform.rotation.x = quat[0]
        t2.transform.rotation.y = quat[1]
        t2.transform.rotation.z = quat[2]
        t2.transform.rotation.w = quat[3]
        self.br.sendTransform(t2)
        self.get_logger().info("Published static TF objectPushT → objectPushT_MuJoCo")

    # ------------------------------------------------------------------
    #  JIT step
    # ------------------------------------------------------------------

    def _make_jitted_step(self, ctrl):
        def step(mjx_data, policy_params, q, dq, mocap_pos, mocap_quat, curr_time):
            mjx_data = mjx_data.replace(
                qpos=jnp.array(q),
                qvel=jnp.array(dq),
                mocap_pos=jnp.array(mocap_pos),
                mocap_quat=jnp.array(mocap_quat),
                time=curr_time,
            )
            new_policy_params, rollouts = ctrl.optimize(mjx_data, policy_params)
            return mjx_data, new_policy_params, rollouts

        return jax.jit(step, donate_argnums=(1,))

    # ------------------------------------------------------------------
    #  Observation helpers
    # ------------------------------------------------------------------

    def _update_T(self):
        if not self.tf_buffer.can_transform(
            "fr3_link0",
            "objectPushT_MuJoCo",
            rclpy.time.Time(),
            rclpy.duration.Duration(seconds=0.0),
        ):
            self.get_logger().warn("TF transform not available yet.")
            return None, None
        tf = self.tf_buffer.lookup_transform(
            "fr3_link0",
            "objectPushT_MuJoCo",
            rclpy.time.Time(),
        )
        lin = np.array(
            [
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ],
            dtype=np.float32,
        )
        quat = np.array(
            [
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            ],
            dtype=np.float32,
        )
        return lin, quat

    def _get_robot_state_np(self):
        robot_q = np.array(
            copy.deepcopy(self._current_joint_state.position),
            dtype=np.float32,
        )
        robot_dq = np.array(
            copy.deepcopy(self._current_joint_state.velocity),
            dtype=np.float32,
        )
        return robot_q, robot_dq

    def _estimate_block_velocity(self, lin_t, quat_t, current_time):
        """Finite-difference velocity from consecutive MoCap frames."""
        vel = np.zeros(6, dtype=np.float32)
        if self._prev_lin_t is not None and self._prev_obs_time is not None:
            dt = current_time - self._prev_obs_time
            if dt > 1e-6:
                vel[0:3] = (lin_t - self._prev_lin_t) / dt
                vel[3:6] = quat_to_angular_velocity(self._prev_quat_t, quat_t, dt)
                vel[0:3] = np.clip(vel[0:3], -2.0, 2.0)
                vel[3:6] = np.clip(vel[3:6], -10.0, 10.0)
        self._prev_lin_t = lin_t.copy()
        self._prev_quat_t = quat_t.copy()
        self._prev_obs_time = current_time
        return vel

    # ------------------------------------------------------------------
    #  Write observed state into debug_data (CPU MuJoCo)
    # ------------------------------------------------------------------

    def _write_debug_state(
        self, lin_t, quat_t, robot_q, robot_dq, block_vel = None, current_time = None
    ):
        """Set debug_data.qpos / qvel from real observations."""
        self.debug_data.qpos[0:7] = np.array(
            [
                lin_t[0],
                lin_t[1],
                lin_t[2],
                quat_t[3],
                quat_t[0],
                quat_t[1],
                quat_t[2],  # wxyz
            ]
        )
        self.debug_data.qpos[7:14] = robot_q
        # self.debug_data.qvel[0:6] = block_vel
        self.debug_data.qvel[6:13] = robot_dq
        self.debug_data.time = current_time

    # ------------------------------------------------------------------
    #  Publish spline
    # ------------------------------------------------------------------

    def _publish_spline(self, actions, dt):
        """Pack [dt, vx0, vy0, vx1, vy1, …] into a Float32MultiArray."""
        msg = Float32MultiArray()
        horizon, nu = actions.shape
        msg.layout.dim = [
            MultiArrayDimension(
                label="meta_and_actions", size=1 + horizon * nu, stride=1
            ),
        ]
        flat = actions.flatten().tolist()
        msg.data = [float(dt)] + flat
        self.spline_pub.publish(msg)
        # self.get_logger().info(f"Published new spline with actions={actions}")

    # ------------------------------------------------------------------
    #  Planning loop (dedicated thread)
    # ------------------------------------------------------------------

    def _planning_loop(self):
        period = 1.0 / self.plan_freq
        self.get_logger().info(
            f"Planning thread started at {self.plan_freq} Hz (period = {period:.3f} s)"
        )
        while not self.shutdown_flag.is_set():
            t_loop = time.time()
            try:
                self._run_controller()
            except Exception as e:
                self.get_logger().error(f"Planning exception: {e}")
                import traceback

                traceback.print_exc()
            elapsed = time.time() - t_loop
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                self.get_logger().warn(
                    f"Planning overran by {-remaining:.3f} s "
                    f"(took {elapsed:.3f} s, budget {period:.3f} s)"
                )

    def _run_controller(self):
        """Single planning iteration.

        Flow (mirrors sim-to-sim ordering):
          1. Read observations
          2. Estimate block velocity
          3. Write state into debug_data
          4. Update ghost mocap bodies from previous predictions
          5. mj_forward (kinematics only)
          6. GPU planning (JAX releases GIL)
          7. Publish spline
          8. Visualise
        """
        t0 = time.time()

        # 1 — observations
        lin_t, quat_t = self._update_T()
        robot_q, robot_dq = self._get_robot_state_np()
        if lin_t is None or quat_t is None:
            self.get_logger().warn("Skipping: missing object state.")
            return
        if robot_q is None or robot_dq is None:
            self.get_logger().warn("Skipping: missing robot state.")
            return
        current_time = self.get_clock().now().nanoseconds / 1e9

        # 2 — block velocity
        # block_vel = self._estimate_block_velocity(lin_t, quat_t, current_time)

        # 3 — write into debug_data
        self._write_debug_state(
            lin_t, quat_t, robot_q, robot_dq, block_vel=None, current_time=current_time
        )

        # 4 — ghost mocap from previous predictions
        distances = np.zeros(self.ctrl.num_randomizations, dtype=np.float32)
        weights = np.ones(self.ctrl.num_randomizations, dtype=np.float32) / self.ctrl.num_randomizations
        if self.predicted_states is not None:
            ref_site = self.predicted_states[:, -1, ...]  # (domains, 7), only for last sight
            mocap_bids = self.mocap_T_bids[: self.ctrl.num_randomizations]
            for idx, bid in enumerate(mocap_bids):
                self.debug_data.mocap_pos[bid] = ref_site[idx, :3]
                self.debug_data.mocap_quat[bid] = ref_site[idx, 3:]
            if len(self.mocap_T_bids) < 10:
                for bid in range(len(self.mocap_T_bids), 10):
                    self.debug_data.mocap_pos[bid] = self.debug_data.xpos[bid]
                    self.debug_data.mocap_quat[bid] = self.debug_data.xquat[bid]

            current_obs = np.array(self.debug_data.qpos[:7], dtype=np.float32)
            distances = jax.vmap(se3_left_invariant_metric, in_axes=(0, None))(
                self.predicted_states[:, -1, ...],
                current_obs,
            )
            if not jnp.allclose(distances, distances[0],rtol=0.01):
                temperature = 0.05 #np.median(distances)/6
                probs = jnp.exp(-distances / temperature)  # temperature scaling
                probs = probs / (jnp.sum(probs) + 1e-12)
                # print("Domain probabilities before update:", probs)
                entropy = -jnp.sum(probs * jnp.log(probs + 1e-12))
                # print("Domain distribution entropy:", entropy)
                max_entropy = jnp.log(len(probs) + 1e-12)
                # print("Max entropy:", max_entropy)
                normalized_entropy = entropy / (max_entropy + 1e-12)
                # print("Normalized entropy:", normalized_entropy)

                new_probs = (
                    1 - normalized_entropy
                ) * probs + normalized_entropy * self.policy_params.domain_weights
                # self.policy_params = self.policy_params.replace(
                #     domain_weights=jnp.array(new_probs)
                # )

                self.get_logger().info(f"Step {self.ctr}: domain error: {distances} | updated weights: {new_probs} | sum of weights: {jnp.sum(new_probs):.4f} ")
            else:
                self.get_logger().info(f"Step {self.ctr}: weights unchanged.")
        
        # 5 — kinematics only (NOT mj_step)
        mujoco.mj_forward(self.debug_model, self.debug_data)

        # 6 — GPU planning
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
        t2 = time.time()

        actions = np.array(self.policy_params.spline)
        self.predicted_states = np.array(self.policy_params.predicted_state)
        self.domain_weights = np.array(self.policy_params.domain_weights)

        # 7 — publish
        self._publish_spline(actions, self.ctrl.task.dt)

        # bookkeeping
        terminal_error = self.ctrl.task.running_cost(self.debug_data)
        ee_position = self.ctrl.task._get_ee_position(self.debug_data)
        self.ctr += 1
        self.log.append(
            {
                "time": current_time - self.start_time,
                "planning_time": t2 - t1,
                "object_pos": lin_t.tolist(),
                "object_quat": quat_t.tolist(),
                "ee_pos": ee_position.tolist(),
                "terminal_error": float(terminal_error),
                "action": actions[0].tolist(),
                # "block_vel": block_vel.tolist(),
                "distances": distances.tolist(),
                "domain_weights": self.domain_weights.tolist(),
            }
        )

        # 8 — visualise rollout traces
        ii = 0
        for k in [2]:  # ee_site
            for i in self.trace_idxs:
                for j in range(self.ctrl.task.planning_horizon):
                    geom = self.viewer.user_scn.geoms[ii]
                    mujoco.mjv_connector(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        0.5,
                        rollouts.trace_sites[0, i, j, k, :3],
                        rollouts.trace_sites[0, i, j + 1, k, :3],
                    )
                    ii += 1
        self.viewer.sync()

        t3 = time.time()
        self.get_logger().info(
            f"Plan: {t2 - t1:.3f} s | State: {t1 - t0:.3f} s | "
            f"Vis: {t3 - t2:.3f} s | Error: {terminal_error:.4f}"
        )

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    def _states_received(self):
        if self._current_joint_state is None:
            self.get_logger().warn("Current joint state not received yet.")
            return False
        self.get_logger().info("Current joint state received.")
        return True

    def _timeout_callback(self):
        self.get_logger().info("Timeout reached, shutting down …")
        self.shutdown_flag.set()
        rclpy.shutdown()

    def save_log(self, path, filename="fr3_dr_real_"):
        log_file = path / f"{filename}{self.ctrl.__class__.__name__}_seed{self.seed}_bad2"
        with open(str(log_file) + ".pkl", "wb") as f:
            pickle.dump(self.log, f)

    def signal_handler(self, sig, frame):
        self.get_logger().info("SIGINT received")
        self.shutdown_flag.set()


# ======================================================================
#  Main
# ======================================================================

if __name__ == "__main__":
    rclpy.init()

    parser = argparse.ArgumentParser(
        description="SMPC Planner Node — publishes action splines.",
    )
    subparsers = parser.add_subparsers(
        dest="algorithm",
        help="Sampling algorithm",
    )
    subparsers.add_parser("mppi")
    subparsers.add_parser("mtp")
    subparsers.add_parser("cem")
    subparsers.add_parser("ps")

    parser.add_argument(
        "--risk",
        choices=["average", "expectation", "var", "cvar", "ivar", "icvar"],
        help="Risk aggregation method for domain randomisation",
    )
   
    args = parser.parse_args()

    # ---- hyper-parameters ----
    seed = 10
    num_samples = 150
    NUM_RANDOMIZATIONS = 10
    planning_freq = 5  # Hz
    max_speed = 0.3

    # ---- risk strategy ----
    risk_alpha = 0.5
    uniform_weights = (
        jnp.ones((NUM_RANDOMIZATIONS,), dtype=jnp.float32) / NUM_RANDOMIZATIONS
    )
    if args.risk is None or args.risk == "average":
        args.risk = "average"
        aggregation = ExpectedCost(weights=uniform_weights) #!
    elif args.risk == "expectation":
        aggregation = ExpectedCost(weights=uniform_weights)
    elif args.risk == "var":
        aggregation = ValueAtRisk(alpha=risk_alpha)
    elif args.risk == "cvar":
        aggregation = ConditionalValueAtRisk(alpha=risk_alpha, weights=uniform_weights)
    elif args.risk == "ivar":
        aggregation = InverseValueAtRisk(alpha=risk_alpha)
    elif args.risk == "icvar":
        aggregation = InverseConditionalValueAtRisk(
            alpha=risk_alpha,
            weights=uniform_weights,
        )
    else:
        raise ValueError(f"Unknown risk strategy: {args.risk}")

    # ---- task ----
    task = PushTFranka(
        ik_type="pinv",
        planning_horizon=14, # was 9
        sim_steps_per_control_step=2,
        ctrl_limits={
            "u_min": jnp.array([-max_speed, -max_speed]),
            "u_max": jnp.array([max_speed, max_speed]),
        },
        trace_sites=["T_1", "T_2", "ee_site", "T_3", "block_site"],
        actuation_type="velocity",
        sampling_space="velocity",
        block_type="dr-free",
    )

    # ---- controller ----
    if args.algorithm is None or args.algorithm == "mtp":
        args.algorithm = "mtp"
        print("Running MTP")
        ctrl = MTP(
            task,
            num_samples=num_samples,
            temperature=0.1,
            M=3,
            N=32,
            sigma_min=0.15,
            sigma_max=0.55,
            sigma_start=0.2,
            num_elites=24,
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
    elif args.algorithm == "mppi":
        print("Running MPPI")
        ctrl = MPPI(
            task,
            num_samples=num_samples,
            alpha=0.1,
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
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    model = task.mj_model
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as v:
        num_traces = 0
        trace_idxs = np.linspace(
            0,
            num_samples - 1,
            num=num_traces,
            dtype=int,
        ).tolist()
        print("Trace indices:", trace_idxs)

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

        node = SMPCPlannerNode(
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
        executor.add_node(node)

        try:
            executor.spin()
        except KeyboardInterrupt:
            print("Shutting down planner …")
        finally:
            node.shutdown_flag.set()
            executor.shutdown()
            path = get_data_path() / "dr_real"
            node.save_log(path)
            node.destroy_node()
            rclpy.shutdown()
