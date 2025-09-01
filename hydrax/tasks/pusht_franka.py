from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from hydrax.files import get_root_path
from hydrax.task_base import Task

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def mujoco_to_scipy_quat(q):
    return np.array([q[1], q[2], q[3], q[0]])

class PushTFranka(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(
        self, planning_horizon: int = 10, sim_steps_per_control_step: int = 5, 
        nu: int = 2, 
        ctrl_limits = {"u_min": jnp.array([-0.1, -0.1]), "u_max": jnp.array([0.1, 0.1])}
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            (get_root_path() / "models" / "pusht_franka_planar" / "scene_mjx.xml").as_posix()
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["ee_site"],
            nu=nu,
            ctrl_limits=ctrl_limits,
        )

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )
        self.ee_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_pos"
        )
        self.ee_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_quat"
        )
        self.block_global_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position_world"
        )
        
        # special to this task
        self.body_id = self.mj_model.body("ee_frame").id

    def reset(self, seed: int = 0) -> None:
        """Randomize the initial pose of the T-shaped block."""
        # Set the random seed for reproducibility
        np.random.seed(seed)
        mj_model = self.mj_model
        mj_model.opt.timestep = 0.002
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_data = mujoco.MjData(self.mj_model)
        # Randomize the block's position and orientation
        pos_x = np.random.uniform(low=-0.25, high=0.25)
        pos_y = np.random.uniform(low=-0.15, high=0.05)
        angle = np.random.uniform(-np.pi, np.pi)

        # Assuming the block's pose is at the beginning of qpos
        mj_data.qpos[0] = pos_x
        mj_data.qpos[1] = -0.1 + pos_y
        mj_data.qpos[2] = angle
        
        # Joint index range (skip floating base joints if any)
        des_pos = np.array([0.0, 0.25, 0.035])
        des_quat = np.array([0.0, 0.7071, 0.7071, 0.0])  # [w, x, y, z]
        
        j_start = 3  # Adjust based on your model (e.g., 3 if floating base)
        n_joints = 7

        # Joint limits
        joint_limits = np.array([self.model.jnt_range[i] for i in range(j_start, j_start + n_joints)])

        # # Initial guess
        q = np.array([0.0, -np.pi/4, 0.0, -9*np.pi/10, 0.0, 3*np.pi/4, np.pi/4])

        # IK loop parameters
        max_iters = 100
        tolerance = 1e-4
        damping = 100e-3

        for i in range(max_iters):
            # Set current joint state
            mj_data.qpos[j_start:j_start+n_joints] = q
            mujoco.mj_forward(self.mj_model, mj_data)

            # Current EE pose
            current_pos = mj_data.xpos[self.body_id]
            current_quat = mj_data.xquat[self.body_id]

            # Position error
            pos_err = des_pos - current_pos  # shape (3,)

            # Orientation error (quaternion distance -> angular velocity vector)
            r_current = R.from_quat(mujoco_to_scipy_quat(current_quat))
            r_desired = R.from_quat(mujoco_to_scipy_quat(des_quat))

            # Rotation needed to go from current to desired
            r_error = r_desired * r_current.inv()

            # Convert to rotation vector (axis-angle * angle)
            orn_err = r_error.as_rotvec()  # shape (3,)

            # Combined 6D task error
            err = np.concatenate([pos_err, orn_err])  # shape (6,)

            if np.linalg.norm(err) < tolerance:
                print(f"Converged in {i} iterations.")
                break

            # Compute Jacobian of the EE
            J_pos = np.zeros((3, self.mj_model.nv))
            J_rot = np.zeros((3, self.mj_model.nv))
            mujoco.mj_jacBody(self.mj_model, mj_data, J_pos, J_rot, self.body_id)

            # Slice columns corresponding to actuated joints
            J = np.vstack([J_pos[:, j_start:j_start+n_joints], J_rot[:, j_start:j_start+n_joints]])  # shape (6, n_joints)

            # Solve damped least squares: dq = (JᵀJ + λ²I)⁻¹ Jᵀ e
            JTJ = J.T @ J
            H = JTJ + damping * np.eye(n_joints)
            g = J.T @ err
            dq = np.linalg.solve(H, g)

            # Update joint configuration
            q += dq

            # Clamp to joint limits
            for j in range(n_joints):
                low, high = joint_limits[j]
                q[j] = np.clip(q[j], low, high)

        else:
            print("IK did not converge.")

        mj_data.qpos[j_start:j_start+n_joints] = q  # Set the robot's joint positions

        # initial control
        mj_data.ctrl[:] = np.zeros(mj_model.nu)
        return mj_model, mj_data

    ##################################
    ##       Goal Error Terms       ##
    ##################################
    
    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        """ Get the position error of the block relative to a goal position."""
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        """ Get the orientation error of the block relative to a goal orientation."""
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(block_quat, goal_quat)
    
    ##################################
    ##  Safety Penalties / Rewards  ##
    ##################################
    
    def _get_table_collision_err(self, state: mjx.Data) -> jax.Array:
        """Check if robot end effector is colliding with the table."""
        sensor_adr = self.model.sensor_adr[self.ee_position_sensor]
        # Get the end effector position
        # Assuming the end effector position is given by a 3D vector in the sensor data
        ee_pos_z = state.sensordata[sensor_adr + 2]  
        # if below table get error, else 0
        table_height = 0.002  # Assuming the table is at z=0
        table_collision = jnp.where(ee_pos_z < table_height, 1.0, 0.0)
        return table_collision  # Return 1 if colliding with table, else 0
        
    def _get_safezone_reward(self, state: mjx.Data) -> jax.Array:
        """Reward for being in a safe zone."""
        sensor_adr = self.model.sensor_adr[self.ee_position_sensor]
        # Get the end effector position
        ee_pos = state.sensordata[sensor_adr : sensor_adr + 3]
        # if z in [-0.001, 0.01]
        safe_zone = jnp.logical_and(ee_pos[2] >= 0.01, ee_pos[2] <= 0.4)
        return jnp.where(safe_zone, -1.0, 0.0)  # Reward of 1 if in safe zone, else 0
         
    
    
    
    ################################## 
    ##      End Effector Terms      ##
    ##################################
    
    def _get_ee_block_distance(self, state: mjx.Data) -> jax.Array:
        """Get the distance between the end effector and the block."""
        sensor_adr = self.model.sensor_adr[self.ee_position_sensor]
        ee_pos = state.sensordata[sensor_adr : sensor_adr + 3]
        
        sensor_adr_block = self.model.sensor_adr[self.block_global_position_sensor]
        block_pos = state.sensordata[sensor_adr_block : sensor_adr_block + 3]
        # Calculate the Euclidean distance between the end effector and the block        
        return jnp.linalg.norm(ee_pos[:2] - block_pos[:2]) # only x,y

        
    def _get_ee_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Get the end effector orientation error."""
        sensor_adr = self.model.sensor_adr[self.ee_orientation_sensor]
        # Get the end effector orientation quaternion
        # Assuming the end effector orientation is given by a quaternion
        # in the sensor data   
        ee_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([0.0, 0.7071, 0.7071, 0.0])  # Assuming goal orientation is aligned with x-axis
        return mjx._src.math.quat_sub(ee_quat, goal_quat)
    

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        
        # Goal error terms 
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        
        total_goal_err = 10 * position_cost + 2 * orientation_cost
        
        
        # This seems to lead to a behavior where the ee doesnt watn tto be in contact, as this increases error
        ee_orientation_err = self._get_ee_orientation_err(state)
        # ee_orientation_cost = jnp.sum(jnp.square(ee_orientation_err))
        
        # safety based
        ee_block_distance = self._get_ee_block_distance(state)
        ee_block_distance_cost = jnp.square(ee_block_distance)
        
        # TODO for velocity a control error makes sense
        control_cost = jnp.sum(jnp.square(control))  # penalize large control inputs

        error = 0.5 * ee_block_distance_cost + total_goal_err + 0.75 * control_cost
        
        return error 
                                                                              

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    ################################## 
    ##            Special           ##
    ##################################
    
    
    
    def make_control_mapper(self):
        j_start = 3
        n_joints = 7
        idx = jnp.array([0, 1, 3, 4, 5])  # x, y, roll, pitch, yaw

        model = self.model
        body_id = self.body_id
        act_min = self.act_min
        act_max = self.act_max

        # Reuse a template data; replace qpos each call (keeps purity for JAX)
        data0 = mjx.make_data(model)

        # FK as a pure function of qpos
        def fk_fn(qpos):
            d = data0.replace(qpos=qpos)
            d = mjx.forward(model, d)
            pos = d.xpos[body_id]  # (3,)
            # Use the private quat->axis/angle you prefer
            axis, angle = mjx._src.math.quat_to_axis_angle(d.xquat[body_id])
            rotvec = angle * axis  # (3,)
            return jnp.concatenate([pos[:2], rotvec], axis=0)  # (5,) ignore z

        # Small output (6) => reverse-mode Jacobian is efficient
        fk_jac = jax.jit(jax.jacrev(fk_fn))

        @jax.jit
        def ik_mapper_transpose(data: mjx.Data, control: jax.Array) -> jax.Array:
            qpos = data.qpos
            
            J = fk_jac(qpos)  # (6, nq)
            # Select task rows and the joint columns we actuate
            J_xy = J[idx, j_start : j_start + n_joints]  # (5, n_joints)
            # Control only for x,y; pad the rest to match selected rows
            adapted_control = jnp.concatenate(
                [control[:2], jnp.zeros(3, dtype=control.dtype)], axis=0
            )  # (5,)

            dq = J_xy.T @ adapted_control                 # (n_joints,)
            dq = jnp.clip(dq, act_min, act_max)
            return dq

        return ik_mapper_transpose
    
    
    
    def ik_mapper_transpose(
        self, model: mjx.Model, data: mjx.Data, control: jax.Array
    ) -> jax.Array:
        
        j_start = 3  
        n_joints = 7
        q = data.qpos
        
        # Pass q_base into fk_fn explicitly
        def fk_fn(qpos):
            data = mjx.make_data(model)
            data = data.replace(qpos=qpos)
            data = mjx.forward(model, data)
            # return end effector position and orientation
            axis, angle = mjx._src.math.quat_to_axis_angle(data.xquat[self.body_id])
            test = angle * axis
            print(f"Axis shape: {test.shape}")
            print(f"xpos shape: {data.xpos[self.body_id].shape}")
            ee_pose = jnp.hstack([
                data.xpos[self.body_id],
                angle * axis
            ])
            print(f"EE pose shape: {ee_pose.shape}")
            return ee_pose

        J = jax.jacobian(lambda qpos: fk_fn(qpos))(q)
        print(f"Jacobian shape: {J.shape}, control shape: {control.shape}")

        idx = [0, 1, 3, 4, 5] # Indices for x, y, roll, pitch, yaw
        J_xy = J[:idx, j_start:j_start+n_joints]  # Jacobian for x, y, roll, pitch, yaw
        
        adapted_control = jnp.concatenate([control[:2], jnp.zeros(3)])  
        dq = J_xy.T @ adapted_control
        dq = jnp.clip(dq, self.act_min, self.act_max)
        return dq
    
    
    def ik_mapper_2d(
        self, model: mjx.Model, data: mjx.Data, control: jax.Array
    ) -> jax.Array:
        
        q = data.qpos
        dq_curr = data.qvel

        # Pass q_base into fk_fn explicitly
        def fk_fn(qpos):
            data = mjx.make_data(model)
            data = data.replace(qpos=qpos)
            data = mjx.forward(model, data)
            # return end effector position and orientation
            axis, angle = mjx._src.math.quat_to_axis_angle(data.xquat[self.body_id])
            test = angle * axis
            print(f"Axis shape: {test.shape}")
            print(f"xpos shape: {data.xpos[self.body_id].shape}")
            ee_pose = jnp.hstack([
                data.xpos[self.body_id],
                angle * axis
            ])
            print(f"EE pose shape: {ee_pose.shape}")
            return ee_pose

        J = jax.jacobian(lambda qpos: fk_fn(qpos))(q)
        print(f"Jacobian shape: {J.shape}, control shape: {control.shape}")
        
        # get deviation in z
        z_pos = data.site_xpos[self.body_id, 2]  # z position of the end effector
        z_vel = J[2, 3:] @ dq_curr[3:]  # z velocity of the end effector
        
        Kp_z = 5.0
        Kd_z = 1.0
        error = z_pos - 0.08
        # half the error if it is positive, to avoid pushing the end effector too high
        jax.lax.cond(error > 0, lambda x: x / 3, lambda x: x, operand=Kp_z)
        # Proportional-Derivative control for z position
        z_vel_feedback = - Kp_z * error - Kd_z * z_vel

        J_full = J[:, 3:]  # Full Jacobian for the robot
        lam = 1e-3
        # J_z_damped_pinv = J_z.T @ jnp.linalg.inv(J_z @ J_z.T + lam * jnp.eye(1))
        # J_xy_damped_pinv = jnp.linalg.pinv(J_xy + lam * jnp.eye(J_xy.shape[0]))
        J_full_damped_pinv = J_full.T @ jnp.linalg.inv(J_full @ J_full.T + lam * jnp.eye(J_full.shape[0]))

        adapted_control = jnp.concatenate([control[:2], jnp.array([z_vel_feedback]), jnp.zeros(J_full.shape[0] - 3)])  # adapt control to 2D
        dq = J_full_damped_pinv @ adapted_control
        dq = jnp.clip(dq, self.act_min, self.act_max)
            
        return dq