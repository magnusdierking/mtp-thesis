from typing import Dict, List, Optional
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

import numpy as np

from hydrax.task_base import Task


def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """
    Compute the foot step height.
    Args:
        amplitude: The height of the step.
        cadence: The cadence of the step (per second).
        duty_ratio: The duty ratio of the step (% on the ground).
        phases: The phase of the step. Warps around 1. (N-dim where N is the number of legs)
        time: The time of the step.
    """

    def step_height(t, footphase, duty_ratio):
        angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
        angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
        value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
        final_value = jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)
        return final_value

    h_steps = amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
        time * 2 * jnp.pi * cadence + jnp.pi,
        2 * jnp.pi * phases,
        duty_ratio,
    )
    return h_steps




class Go2VelocityTask(Task):
    """
    Simple locomotion task for a Unitree Go2-like quadruped using IMU + joint sensors.

    Objective (tunable):
      - Track a target body forward velocity in +x (optionally y, yaw rate).
      - Keep body upright (penalize orientation error) and maintain target height.
      - Regularize joint velocities and control magnitudes.

    This follows the same Task structure as the PushTFranka example (sensors via
    `self.model.sensor_adr[...]`, cost in `running_cost`, and optional reset),
    but acts directly in joint space, so the control mapper is the identity.
    """

    # Joint order used for control and per-joint sensors
    JOINT_ORDER: List[str] = [
        # Front Left
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        # Front Right
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        # Rear  Left
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        # Rear  Right
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    # Map each joint to its corresponding position & velocity sensor names from the user's XML
    SENSOR_MAP_POS: Dict[str, str] = {
        "FL_hip_joint": "abduction_front_left_pos",
        "FL_thigh_joint": "hip_front_left_pos",
        "FL_calf_joint": "knee_front_left_pos",
        "FR_hip_joint": "abduction_front_right_pos",
        "FR_thigh_joint": "hip_front_right_pos",
        "FR_calf_joint": "knee_front_right_pos",
        "RL_hip_joint": "abduction_hind_left_pos",
        "RL_thigh_joint": "hip_hind_left_pos",
        "RL_calf_joint": "knee_hind_left_pos",
        "RR_hip_joint": "abduction_hind_right_pos",
        "RR_thigh_joint": "hip_hind_right_pos",
        "RR_calf_joint": "knee_hind_right_pos",
    }

    SENSOR_MAP_VEL: Dict[str, str] = {
        "FL_hip_joint": "abduction_front_left_vel",
        "FL_thigh_joint": "hip_front_left_vel",
        "FL_calf_joint": "knee_front_left_vel",
        "FR_hip_joint": "abduction_front_right_vel",
        "FR_thigh_joint": "hip_front_right_vel",
        "FR_calf_joint": "knee_front_right_vel",
        "RL_hip_joint": "abduction_hind_left_vel",
        "RL_thigh_joint": "hip_hind_left_vel",
        "RL_calf_joint": "knee_hind_left_vel",
        "RR_hip_joint": "abduction_hind_right_vel",
        "RR_thigh_joint": "hip_hind_right_vel",
        "RR_calf_joint": "knee_hind_right_vel",
    }

    def __init__(
        self,
        xml_path: Optional[str | Path],
        planning_horizon: int = 10,
        sim_steps_per_control_step: int = 5,
        actuation_type: str = "position", # "velocity" or "position"
        target_vx: float = 0.0, # m/s
        target_vy: float = 0.0, # m/s
        target_yaw_rate: float = 0.0, # rad/s
        target_height: Optional[float] = 0.445, # meters; None -> no height term
        trace_sites: List[str] = ["imu"],
        ctrl_limits: Optional[Dict[str, jnp.ndarray]] = None,
        home_keyframe: Optional[str] = "home",
        ):
        
        if xml_path is None:
            raise ValueError("Please provide the path to your Go2 MuJoCo XML (scene_mjx.xml)")
        xml_path = Path(xml_path).as_posix()
        mj_model = mujoco.MjModel.from_xml_path(xml_path)


        # Cache optional home keyframe id (used in reset to avoid spawning underground)
        self._home_kf_id = -1
        if home_keyframe:
            try:
                self._home_kf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, home_keyframe)
            except Exception:
             self._home_kf_id = -1


        # Control dimension equals the number of actuated joints (12)
        nu = len(self.JOINT_ORDER)


        # Default symmetric limits for joint-velocity control (tune as needed)
        if ctrl_limits is None:
            ctrl_limits = {
            "u_min": jnp.array([-2.0] * nu), # rad/s or equivalent position rate
            "u_max": jnp.array([ 2.0] * nu),
            }


        super().__init__(
        mj_model,
        planning_horizon=planning_horizon,
        sim_steps_per_control_step=sim_steps_per_control_step,
        trace_sites=trace_sites,
        nu=nu,
        ctrl_limits=ctrl_limits,
        )

        # ------ Cache joint & sensor ids from mj_model ------
        # Joint indices for control and joint limits
        self.actuator_joint_names = list(self.JOINT_ORDER)
        self.actuator_joint_idxs = [mj_model.joint(n).id-1 for n in self.actuator_joint_names]
        print(self.actuator_joint_idxs)
        self.joint_limits = self.mj_model.jnt_range[self.actuator_joint_idxs]

        # Build mapping from our JOINT_ORDER to actuator indices so we can set ctrl correctly
        trn = np.array(mj_model.actuator_trnid).reshape(mj_model.nu, 2)
        self._actuator_index_for_joint: List[int] = []
        for jidx in self.actuator_joint_idxs:
            candidates = np.nonzero(trn[:, 0] == jidx)[0]
        if len(candidates) == 0:
            raise ValueError(f"No actuator targets joint id {jidx} ({mj_model.joint(jidx).name}).")
        self._actuator_index_for_joint.append(int(candidates[0]))
        print(f"Actuator indices for joints: {self._actuator_index_for_joint}")


        # ------ Task targets & weights ------
        self.target_vx = float(target_vx)
        self.target_vy = float(target_vy)
        self.target_yaw_rate = float(target_yaw_rate)
        self.target_height = None if target_height is None else float(target_height)


        # Quaternion upright (world-aligned): [w, x, y, z]
        self.goal_quat_body = jnp.array([1.0, 0.0, 0.0, 0.0])


        # Weights (tune to your liking)
        self.w_vel = jnp.array([4.0, 2.0]) # vx, vy tracking
        self.w_yaw = 1.5
        self.w_orient = 0.5
        self.w_height = 10.0
        self.w_joint_vel = 1e-3
        self.w_control = 1e-4
        self.w_gait = 10.0


        # Per-joint sensor ids (scalar sensors)
        self._jpos_sensor_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, self.SENSOR_MAP_POS[n])
        for n in self.JOINT_ORDER
        ]
        self._jvel_sensor_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, self.SENSOR_MAP_VEL[n])
        for n in self.JOINT_ORDER
        ]


        # IMU / body-frame sensors (vector sensors)
        self._imu_orientation_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )
        self._imu_global_pos_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "global_position"
        )
        self._imu_global_linvel_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "global_linvel"
        )
        self._imu_global_angvel_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "global_angvel"
        )


        # Actuation mode
        if actuation_type not in ("velocity", "position"):
            raise ValueError("actuation_type must be 'velocity' or 'position'")
        self.actuation_type = actuation_type

    # -------------------------- Reset / Initialization --------------------------
    def reset(self, seed: int = 0):
        """Reset to a known-good pose. If a keyframe named `home_keyframe` exists,
        use it. Otherwise fall back to a crouched stance with base z above ground.
        This prevents the robot from spawning inside the terrain."""
        np.random.seed(seed)


        mj_model = self.mj_model
        mj_data = mujoco.MjData(mj_model)


        if getattr(self, "_home_kf_id", -1) >= 0:
            # Use the XML keyframe for a clean start (qpos, qvel, act, ctrl if available)
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, self._home_kf_id)
            # Ensure ctrl is aligned with our JOINT_ORDER if we're in position mode
            if self.actuation_type == "position":
                try:
                    k = self._home_kf_id
                    key_ctrl = np.array(mj_model.key_ctrl[k * mj_model.nu : (k + 1) * mj_model.nu])
                    if key_ctrl.size == mj_model.nu:
                        mj_data.ctrl[:] = key_ctrl
                    else:
                        # Fallback to joint positions as targets
                        q = np.array([mj_data.qpos[adr] for adr in self._jnt_qadr_for_joint])
                        for i, aidx in enumerate(self._actuator_index_for_joint):
                            mj_data.ctrl[aidx] = q[i]
                except Exception:
                    q = np.array([mj_data.qpos[adr] for adr in self._jnt_qadr_for_joint])
                    for i, aidx in enumerate(self._actuator_index_for_joint):
                        mj_data.ctrl[aidx] = q[i]
            else:
                mj_data.ctrl[:] = mj_model.key_ctrl[0:mj_model.nu]
        else:
            # Fallback: place base above ground and set a stable crouch
            # Root: [x, y, z, qw, qx, qy, qz]
            mj_data.qpos[0:7] = np.array([0.0, 0.0, 0.67, 1.0, 0.0, 0.0, 0.0])

            # Crouched legs per JOINT_ORDER: abduction, thigh, knee
            default_leg = np.array([0.0, 0.9, -1.8])
            q = np.tile(default_leg, 4) + 0.02 * np.random.randn(12)
            # Clamp to joint limits and write using joint qpos addresses
            for j, (jidx, adr) in enumerate(zip(self.actuator_joint_idxs, self._jnt_qadr_for_joint)):
                low, high = self.joint_limits[j]
            mj_data.qpos[adr] = float(np.clip(q[j], low, high))


            if self.actuation_type == "position":
                for i, (adr, aidx) in enumerate(zip(self._jnt_qadr_for_joint, self._actuator_index_for_joint)):
                 mj_data.ctrl[aidx] = mj_data.qpos[adr]
            else:
                mj_data.ctrl[:] = 0.0


        # Finalize
        mujoco.mj_forward(mj_model, mj_data)
        return mj_model, mj_data

    # ------------------------------ Sensor helpers ------------------------------
    @property
    def _jpos_adrs(self):
        return jnp.array([self.model.sensor_adr[sid] for sid in self._jpos_sensor_ids], dtype=jnp.int32)

    @property
    def _jvel_adrs(self):
        return jnp.array([self.model.sensor_adr[sid] for sid in self._jvel_sensor_ids], dtype=jnp.int32)

    def _get_joint_pos(self, state: mjx.Data) -> jax.Array:
        # Each jointpos sensor contributes a single channel
        return jnp.take(state.sensordata, self._jpos_adrs)

    def _get_joint_vel(self, state: mjx.Data) -> jax.Array:
        return jnp.take(state.sensordata, self._jvel_adrs)

    def _get_body_quat(self, state: mjx.Data) -> jax.Array:
        adr = self.model.sensor_adr[self._imu_orientation_id]
        return state.sensordata[adr : adr + 4]

    def _get_body_pos(self, state: mjx.Data) -> jax.Array:
        adr = self.model.sensor_adr[self._imu_global_pos_id]
        return state.sensordata[adr : adr + 3]

    def _get_body_linvel(self, state: mjx.Data) -> jax.Array:
        adr = self.model.sensor_adr[self._imu_global_linvel_id]
        return state.sensordata[adr : adr + 3]

    def _get_body_angvel(self, state: mjx.Data) -> jax.Array:
        adr = self.model.sensor_adr[self._imu_global_angvel_id]
        return state.sensordata[adr : adr + 3]
    
    def _get_feet_height(self, state: mjx.Data) -> jax.Array:   
        # Example: get foot heights from site positions
        foot_heights = []
        for foot_name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
            site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, foot_name)
            foot_pos = self.mj_model.site_pos[site_id]
            foot_heights.append(foot_pos[2])  # z-coordinate
        return jnp.array(foot_heights)

    # ------------------------------ Cost functions ------------------------------
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        # Velocity tracking (x & y in world frame)
        v = self._get_body_linvel(state)
        vel_err = jnp.stack([v[0] - self.target_vx, v[1] - self.target_vy])
        vel_cost = jnp.sum(self.w_vel * (vel_err ** 2))

        # Prefer zero yaw rate
        wz = self._get_body_angvel(state)[2]
        yaw_cost = self.w_yaw * ((wz - self.target_yaw_rate) ** 2)

        # Upright body orientation (quat distance to identity/upright)
        body_q = self._get_body_quat(state)
        orient_err = mjx._src.math.quat_sub(body_q, self.goal_quat_body)
        orient_cost = self.w_orient * jnp.sum(orient_err ** 2)

        # Height tracking (optional)
        height_cost = 0.0
        if self.target_height is not None:
            z = self._get_body_pos(state)[2]
            height_cost = self.w_height * ((z - self.target_height) ** 2)
            
        # gait via foot heights
        foot_heights = self._get_feet_height(state)
        foot_targets = get_foot_step(
            duty_ratio=0.75,
            cadence=1.0,
            amplitude=0.08,
            phases=jnp.array([0.0, 0.5, 0.5, 0.0]),
            time=state.time,
        )
        gait_cost = self.w_gait * jnp.sum((foot_heights - foot_targets) ** 2)
        
        # Smoothness regularizers
        jvel = self._get_joint_vel(state)
        joint_vel_cost = self.w_joint_vel * jnp.sum(jvel ** 2)
        ctrl_cost = self.w_control * jnp.sum(control ** 2)

        return vel_cost + yaw_cost + orient_cost + height_cost + joint_vel_cost + ctrl_cost + gait_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return self.running_cost(state, jnp.zeros(self.model.nu))


    # --------------------------- (Optional) Randomization ---------------------------
    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        # Example: randomize geom friction slightly
        # n_geoms = self.model.geom_friction.shape[0]
        # multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.9, maxval=1.1)
        # new_frictions = self.model.geom_friction.at[:, 0].set(
        #     self.model.geom_friction[:, 0] * multiplier
        # )
        # return {"geom_friction": new_frictions}
        return {}

################################## 
##            Special           ##
##################################


    def make_gravity_compensator(self):
        return None

    def make_control_mapper(self):
        return None