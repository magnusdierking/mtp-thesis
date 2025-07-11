from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from hydrax.files import get_root_path
from hydrax.task_base import Task


class Dummy(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(
        self, planning_horizon: int = 15, sim_steps_per_control_step: int = 10
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            (get_root_path() / "hydrax" / "models" / "dummy" / "scene.xml").as_posix()
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["ee_site"],
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
    ##          For Testing         ##
    ##################################

    def _get_dummy_error(self, state: mjx.Data) -> jax.Array:
        # simply push back to initial position
        initial_position = jnp.array([0.0, -np.pi/4, 0.0, -9*np.pi/10, 0.0, 3*np.pi/4, np.pi/4]) 
        # get the current joint positions
        current_position = state.qpos[3:]  # Assuming the first 3 are the position of the block
        # calculate the error
        return jnp.linalg.norm(current_position - initial_position)


    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        dummy_error = self._get_dummy_error(state)
        dummy_cost = jnp.sum(jnp.square(dummy_error))
        
        return dummy_cost                                                                             

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}
