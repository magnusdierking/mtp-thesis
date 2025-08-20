from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from hydrax.files import get_root_path
from hydrax.task_base import Task


class PushT(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(
        self, planning_horizon: int = 25, sim_steps_per_control_step: int = 10
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            (get_root_path() / "hydrax" / "models" / "pusht" / "scene.xml").as_posix()
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["pusher"],
        )

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

    def reset(self, seed: int = 0) -> None:
        """Randomize the initial pose of the T-shaped block."""
        # Set the random seed for reproducibility
        np.random.seed(seed)
        mj_model = self.mj_model
        mj_model.opt.timestep = 0.002
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_data = mujoco.MjData(self.mj_model)
        pos_xy = np.random.uniform(low=-0.3, high=0.3, size=(2,))
        angle = np.random.uniform(-np.pi, np.pi)

        # Assuming the block's pose is at the beginning of qpos
        mj_data.qpos[:2] = pos_xy
        mj_data.qpos[2] = angle
        return mj_model, mj_data

    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(block_quat, goal_quat)

    def _close_to_block_err(self, state: mjx.Data) -> jax.Array:
        block_pos = state.qpos[:2]
        pusher_pos = state.qpos[3:] + jnp.array([0.0, 0.1])
        return block_pos - pusher_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        close_to_block_err = self._close_to_block_err(state)

        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        close_to_block_cost = jnp.sum(jnp.square(close_to_block_err))

        return 5 * position_cost + orientation_cost + 0.01 * close_to_block_cost + 0.01 * jnp.sum(jnp.square(control))

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}
