from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from hydrax.files import get_root_path
from hydrax.task_base import Task


class CubeRotation(Task):
    """Cube rotation with the LEAP hand."""

    def __init__(
        self, planning_horizon: int = 3, sim_steps_per_control_step: int = 4
    ):
        mj_model = mujoco.MjModel.from_xml_path( (get_root_path() / "models" /  "cube" / "scene.xml").as_posix())

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["cube_center", "if_tip", "mf_tip", "rf_tip", "th_tip"],
        )

        self.cube_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_position"
        )
        self.cube_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "cube_orientation"
        )

        self.delta = 0.015
        self.success_threshold = 0.03
        self.goal_orientation = jnp.array([1.0, 0.0, 0.0, 0.0])
        
        nbr_actuators = mj_model.nu
        self.actuator_joint_idxs = np.arange(nbr_actuators).tolist()

    def reset(self, seed: int = 0) -> None:
        """Randomize the target cube orientation, just for experiments."""
        np.random.seed(seed)
        mj_model = self.mj_model
        random_axis = np.random.normal(size=(3,))
        random_axis /= np.linalg.norm(random_axis)
        random_angle = np.random.uniform(0, 2 * jnp.pi)

        qw = np.cos(random_angle / 2)
        qx, qy, qz = random_axis * np.sin(random_angle / 2)
        self.goal_orientation = jnp.array([qw, qx, qy, qz])

        return mj_model, mujoco.MjData(mj_model)

    def success(self, state):
        position_err = self._get_cube_position_err(state)
        orientation_err = self._get_cube_orientation_err(state)
        return (jnp.linalg.norm(position_err) + jnp.linalg.norm(orientation_err)) < self.success_threshold

    def _get_cube_position_err(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.cube_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_cube_orientation_err(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.cube_orientation_sensor]
        cube_quat = state.sensordata[sensor_adr : sensor_adr + 4]

        return mjx._src.math.quat_sub(cube_quat, self.goal_orientation)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        position_err = self._get_cube_position_err(state)
        squared_distance = jnp.sum(jnp.square(position_err[0:2]))
        position_cost = 0.1 * squared_distance + 100 * jnp.maximum(
            squared_distance - self.delta**2, 0.0
        )

        orientation_err = self._get_cube_orientation_err(state)
        orientation_cost = jnp.sum(jnp.square(orientation_err))

        grasp_cost = 0.001 * jnp.sum(jnp.square(control))

        return position_cost + orientation_cost + grasp_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        position_err = self._get_cube_position_err(state)
        orientation_err = self._get_cube_orientation_err(state)
        return 100 * jnp.sum(jnp.square(position_err)) + jnp.sum(jnp.square(orientation_err))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        shift = 0.005 * jax.random.normal(rng, (self.model.nq,))
        return {"qpos": data.qpos + shift}

    def make_control_mapper(self):
        return None
    
    def make_gravity_compensator(self):
        return None