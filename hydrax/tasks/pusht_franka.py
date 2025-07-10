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
        self, planning_horizon: int = 15, sim_steps_per_control_step: int = 10
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
        mj_data.qpos[3:] = np.array([0.0, 
                                    -np.pi/4,
                                    0.0,
                                    -9*np.pi/10,
                                    0.0,
                                    3*np.pi/4,
                                    np.pi/4]) 

        mj_data.ctrl[:] = np.array([0.0, 
                                    -np.pi/4,
                                    0.0,
                                    -9*np.pi/10,
                                    0.0,
                                    3*np.pi/4,
                                    np.pi/4])  # Initialize ctrl to qpos for the first 7 controls
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
        return jnp.linalg.norm(ee_pos - block_pos)

        
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
        
        total_goal_err = 7 * position_cost + 2 * orientation_cost
        
        # This seems to lead to a behavior where the ee doesnt watn tto be in contact, as this increases error
        ee_orientation_err = self._get_ee_orientation_err(state)
        ee_orientation_cost = jnp.sum(jnp.square(ee_orientation_err))
        
        # safety based
        ee_block_distance = self._get_ee_block_distance(state)
        safezone_reward = self._get_safezone_reward(state)
        ee_block_distance_cost = jnp.square(ee_block_distance)# seems to be important for greavity compensation
        safezone_reward = jnp.sum(safezone_reward)
        
        total_safety_err = 0.75 * ee_block_distance_cost + 0.5 * safezone_reward
        
        # sacle up goal error to be at least as important as the safety terms
        scaling = 2 * total_safety_err / (total_goal_err + 1e-6)  # avoid division by zero
        
        # TODO for velocity a control error makes sense
        
        
        error = jax.lax.cond(total_goal_err < total_safety_err,
                             lambda x: scaling * total_goal_err + total_safety_err,
                             lambda x: total_goal_err + total_safety_err,
                             operand=None)
        
        return error + jnp.sum(jnp.square(control)) * 0.01 + ee_orientation_cost * 0.1
                                                                              

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}
