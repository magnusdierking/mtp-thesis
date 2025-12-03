from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        planning_horizon: int,
        sim_steps_per_control_step: int,
        trace_sites: Sequence[str] = [],
        nu: Optional[int] = None,
        ctrl_limits: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            planning_horizon: The number of control steps (T) to plan over.
            sim_steps_per_control_step: The number of simulation steps to take
                                        for each control step.
            trace_sites: A list of site names to visualize with traces.
            nu: The number of control inputs (if not specified, inferred from the model).
            

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
              
        Warning: If the control cimension nu is not specified, it will be inferred
                 from the model. If the specififed nu is different from the model, the algorithm has to 
                 implement a mapping function
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model)
        self.planning_horizon = planning_horizon
        self.sim_steps_per_control_step = sim_steps_per_control_step
        
        # Set actuator limits
        self.act_min = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -jnp.inf,
        )
        self.act_max = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            jnp.inf,
        )
        self.nu = nu if nu is not None else self.model.nu    
        
        if nu is None:
            # Set actuator limits
            self.u_min = self.act_min.astype(jnp.float32)
            self.u_max = self.act_max.astype(jnp.float32)
        else:
            assert ctrl_limits is not None, "Control limits must be provided if nu is specified."
            self.u_min = ctrl_limits.get("u_min", jnp.full(nu, -np.inf, dtype=jnp.float32)).astype(jnp.float32)
            self.u_max = ctrl_limits.get("u_max", jnp.full(nu, np.inf, dtype=jnp.float32)).astype(jnp.float32)
            
        # Timestep for each control step
        self.dt = mj_model.opt.timestep * sim_steps_per_control_step

        # Get site IDs for points we want to trace
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
        )
        self.task_success = False
        self.success_threshold = 5e-2
    
    def success(self, state: mjx.Data) -> bool:
        """Check if the task is successful.

        Args:
            state: The current state xₜ.

        Returns:
            True if the task is successful, False otherwise.
        """
        return True

    def reset(self, seed: int = 0) -> mujoco.MjData:
        """Reset the simulation to a random initial state.
        """
        np.random.seed(seed)
        self.task_success = False
        return self.mj_model, mujoco.MjData(self.mj_model)

    def make_control_mapper(self):
        """Override this method to provide a custom control mapper.
        """
        return None
    
    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 7))
        # combine site positions and orientations
        site_pos = state.site_xpos[self.trace_site_ids]
        site_mat = state.site_xmat[self.trace_site_ids] # 3 x 3
        # convert to quaternion
        def mat2quat(mat):
            # transform 3x3 rotation matrix to quaternion
            w = jnp.sqrt(1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2]) / 2.0
            x = (mat[2, 1] - mat[1, 2]) / (4.0 * w)
            y = (mat[0, 2] - mat[2, 0]) / (4.0 * w)
            z = (mat[1, 0] - mat[0, 1]) / (4.0 * w)
            return jnp.array([w, x, y, z])
        
        site_xquat = jax.vmap(mat2quat)(site_mat)
        return jnp.concatenate([site_pos, site_xquat], axis=-1)  
        # return state.site_xpos[self.trace_site_ids]

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}