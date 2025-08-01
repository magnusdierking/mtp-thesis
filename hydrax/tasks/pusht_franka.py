from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from hydrax.files import get_root_path
from hydrax.task_base import Task


class PushTFranka(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(
        self, planning_horizon: int = 10, sim_steps_per_control_step: int = 10, 
        nu: int = 2, 
        ctrl_limits = {"u_min": jnp.array([-2.0, -2.0]), "u_max": jnp.array([2.0, 2.0])}
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            (get_root_path() / "hydrax" / "models" / "pusht_franka" / "scene.xml").as_posix()
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
        self.body_id = self.mj_model.body("ee_frame").id # The body of the T-shaped block

    def reset(self, seed: int = 0) -> None:
        """Randomize the initial pose of the T-shaped block."""
        # Set the random seed for reproducibility
        np.random.seed(seed)
        mj_model = self.mj_model
        mj_model.opt.timestep = 0.001
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_data = mujoco.MjData(self.mj_model)
        pos_x = np.random.uniform(low=-0.25, high=0.25)
        pos_y = np.random.uniform(low=-0.15, high=0.05)
        angle = np.random.uniform(-np.pi, np.pi)

        # Assuming the block's pose is at the beginning of qpos
        mj_data.qpos[0] = pos_x
        mj_data.qpos[1] = -0.1 + pos_y
        mj_data.qpos[2] = angle
        
        # set the initial joint angles of the robot
        # mj_data.qpos[3:] = np.array([0.0, 
        #                             -np.pi/4,
        #                             0.0,
        #                             -9*np.pi/10,
        #                             0.0,
        #                             3*np.pi/4,
        #                             np.pi/4]) 
        mj_data.qpos[3:] = np.array([0.0,
                                    -0.145,
                                    0.0,
                                    -2.43,
                                    0.0,
                                    2.36,
                                    0.78])
        # mj_data.ctrl[:] = np.array([0.0, 
        #                             -np.pi/4,
        #                             0.0,
        #                             -9*np.pi/10,
        #                             0.0,
        #                             3*np.pi/4,
        #                             np.pi/4])  # Initialize ctrl to qpos for the first 7 controls
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
    

    ################################## 
    ##          For Testing         ##
    ##################################

    def _get_dummy_error(self, state: mjx.Data) -> jax.Array:
        # simply push back to initial position
        initial_position = jnp.array([0.0, -np.pi/4, 0.0, -9*np.pi/10, 0.0, 3*np.pi/4, np.pi/4]) 


    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        # goal based
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        
        total_goal_err = 10 * position_cost + 2 * orientation_cost
        
        # This seems to lead to a behavior where the ee doesnt watn tto be in contact, as this increases error
        ee_orientation_err = self._get_ee_orientation_err(state)
        ee_orientation_cost = jnp.sum(jnp.square(ee_orientation_err))
        
        # safety based
        ee_block_distance = self._get_ee_block_distance(state)
        ee_block_distance_cost = jnp.square(ee_block_distance)
        
        # total_safety_err = 0.75 * ee_block_distance_cost + 0.5 * safezone_reward
        
        # sacle up goal error to be at least as important as the safety terms
        # scaling = 2 * total_safety_err / (total_goal_err + 1e-6)  # avoid division by zero
        
        # TODO for velocity a control error makes sense (is already implicitly in the algo)
        
        
        error = 0.5 * ee_block_distance_cost + total_goal_err #+ ee_orientation_cost
        # jax.lax.cond(total_goal_err < total_safety_err,
        #                      lambda x: scaling * total_goal_err + total_safety_err,
        #                      lambda x: total_goal_err + total_safety_err,
        #                      operand=None)
        
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
    
    # @jax.jit
    def ik_mapper(
        self, model: mjx.Model, data: mjx.Data, control: jax.Array
    ) -> jax.Array:
        """Map the end-effector velocity to joint velocities using Jacobian transpose IK."""

        q = data.qpos[3:]
        q_base = data.qpos[:3]  # floating base

        # Pass q_base into fk_fn explicitly
        def fk_fn(qpos, q_base):
            qpos = jnp.asarray(qpos).reshape(-1)
            q_base = jnp.asarray(q_base).reshape(-1)

            full_qpos = jnp.concatenate([q_base, qpos])
            assert full_qpos.ndim == 1, f"Expected 1D qpos, got shape {full_qpos.shape}"
            assert full_qpos.shape[0] == model.nq, f"Expected qpos.shape == {model.nq}, got {full_qpos.shape}"

            data = mjx.make_data(model)
            data = data.replace(qpos=full_qpos)

            return mjx.forward(model, data).xpos[self.body_id]
        
        # Ensure types and shapes are correct
        assert q.shape[0] == model.nv - 3, "Joint positions must match model's degrees of freedom."
        assert q_base.shape[0] == 3, "Base should have 3 DOF."

        J = jax.jacobian(lambda qpos: fk_fn(qpos, q_base))(q)


        dq = J.T @ control
        dq_clipped = jnp.clip(dq, self.act_min, self.act_max)
        return dq_clipped
    
    
    def ik_mapper_2d(
        self, model: mjx.Model, data: mjx.Data, control: jax.Array
    ) -> jax.Array:
        
        q = data.qpos
        dq_curr = data.qvel

        # Pass q_base into fk_fn explicitly
        def fk_fn(qpos):
            data = mjx.make_data(model)
            data = data.replace(qpos=qpos)
            return mjx.forward(model, data).xpos[self.body_id]
        
        J = jax.jacobian(lambda qpos: fk_fn(qpos))(q)
        
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

        
        
        # Extract the relevant parts of the Jacobian for the robot
        # J_xy = J[:2, 3:]
        # J_z = J[2, 3:]
        J_xyz = J[:, 3:]  # Full Jacobian for the robot
        lam = 1e-3
        # J_z_damped_pinv = J_z.T @ jnp.linalg.inv(J_z @ J_z.T + lam * jnp.eye(1))
        # J_xy_damped_pinv = jnp.linalg.pinv(J_xy + lam * jnp.eye(J_xy.shape[0]))
        J_xyz_damped_pinv = J_xyz.T @ jnp.linalg.inv(J_xyz @ J_xyz.T + lam * jnp.eye(J_xyz.shape[0]))
        
        adapted_control = jnp.concatenate([control[:2], jnp.array([z_vel_feedback])])
        dq = J_xyz_damped_pinv @ adapted_control
        dq = jnp.clip(dq, self.act_min, self.act_max)
            
        return dq