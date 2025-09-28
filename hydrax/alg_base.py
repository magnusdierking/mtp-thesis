from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Tuple, Optional, Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.risk import AverageCost, RiskStrategy
from hydrax.task_base import Task


@dataclass
class Trajectory:
    """Data class for storing rollout data.

    Attributes:
        controls: Control actions for each time step (size T).
        costs: Costs associated with each time step (size T+1).
        trace_sites: Positions of trace sites at each time step (size T+1).
    """

    controls: jax.Array
    costs: jax.Array
    trace_sites: jax.Array

    def __len__(self):
        """Return the number of time steps in the trajectory (T)."""
        return self.costs.shape[-1] - 1


class SamplingBasedController(ABC):
    """An abstract sampling-based MPC algorithm interface."""

    def __init__(
        self,
        task: Task,
        num_randomizations: int,
        risk_strategy: RiskStrategy,
        seed: int,
    ):
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
            seed: The random seed for domain randomization.
        """
        self.task = task
        self.num_randomizations = max(num_randomizations, 1)

        # Risk strategy defaults to average cost
        if risk_strategy is None:
            risk_strategy = AverageCost()
        self.risk_strategy = risk_strategy

        # Use a single model (no domain randomization) by default
        self.model = task.model
        self.randomized_axes = None
        
        # Control mapper for applying controls to the model
        self.control_mapper = task.make_control_mapper()
        self.gravity_compensator = task.make_gravity_compensator()

        # Set the random seed for domain randomization
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        if self.num_randomizations > 1:
            # Make domain randomized models
            rng = jax.random.key(seed)
            rng, subrng = jax.random.split(rng)
            subrngs = jax.random.split(subrng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_model)(subrngs)
            self.model = self.task.model.tree_replace(randomizations)

            # Keep track of which elements of the model have randomization
            self.randomized_axes = jax.tree.map(lambda x: None, self.task.model)
            self.randomized_axes = self.randomized_axes.tree_replace(
                {key: 0 for key in randomizations.keys()}
            )

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Sample random control sequences
        controls, params = self.sample_controls(params)
        controls = jnp.clip(controls, self.task.u_min, self.task.u_max)

        # Roll out the control sequences, applying domain randomizations and
        # combining costs using self.risk_strategy.
        rng, dr_rng = jax.random.split(params.rng)
        rollouts = self.rollout_with_randomizations(state, controls, dr_rng)
        params = params.replace(rng=rng)

        # Update the policy parameters based on the combined costs
        params = self.update_params(params, rollouts)
        return params, rollouts

    def rollout_with_randomizations(
        self,
        state: mjx.Data,
        controls: jax.Array,
        rng: jax.Array,
    ) -> Trajectory:
        """Compute rollout costs, applying domain randomizations.

        Args:
            state: The initial state x₀.
            controls: The control sequences, size (num rollouts, horizon - 1).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, costs, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Set the initial state for each rollout.
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        if self.num_randomizations > 1:
            # Randomize the initial states for each domain randomization
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                states, subrngs
            )
            states = states.tree_replace(randomizations)

        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        # _, rollouts = jax.vmap(
        #     self.eval_rollouts, in_axes=(self.randomized_axes, 0, None)
        # )(self.model, states, controls)
        rollouts = jax.vmap(
            self.eval_rollouts, in_axes=(self.randomized_axes, 0, None)
        )(self.model, states, controls)
        
        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        #controls = rollouts.controls[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st
        return rollouts.replace(
            costs=costs, controls=controls, trace_sites=trace_sites
        )

    @partial(jax.vmap, in_axes=(None, None, None, 0)) # 0 to vectorize over num_rollotus
    def eval_rollouts(
        self, model: mjx.Model, state: mjx.Data, controls: jax.Array
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, size (num rollouts, horizon - 1).

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            
            # jax.debug.print("qpos shape: {}, dtype: {}, norm: {}", x.qpos.shape, x.qpos.dtype, jnp.linalg.norm(x.qpos))
            # jax.debug.print("qvel shape: {}, dtype: {}, norm: {}", x.qvel.shape, x.qvel.dtype, jnp.linalg.norm(x.qvel))
            x = mjx.forward(model, x)  # compute site positions
            # jax.debug.print(f"Control shape: {u.shape}")
            u_mapped = self.control_mapper(x, u) if self.control_mapper else u
            
            cost = self.task.dt * self.task.running_cost(x, u_mapped)
            sites = self.task.get_trace_sites(x)
            # jax.debug.print("After running cost and trace sites")

            def _step_debug(_: int, x: mjx.Data):
                if self.gravity_compensator:
                    tau_g = self.gravity_compensator(x)
                    qfrc = jnp.zeros_like(x.qfrc_applied).at[jnp.array(self.task.actuator_joint_idxs)].set(tau_g[jnp.array(self.task.actuator_joint_idxs)])
                    xfrc = jnp.zeros_like(x.xfrc_applied)
                    x = x.replace(qfrc_applied=qfrc, xfrc_applied=xfrc) 
                 
                    # x = x.replace(
                    #     qfrc_applied=x.qfrc_applied.at[:].set(0.0)
                    # )
                    # x = x.replace(
                    #     xfrc_applied=x.xfrc_applied.at[:].set(0.0)
                    # )
                    # x = x.replace(
                    #     qfrc_applied=x.qfrc_applied.at[jnp.array(self.task.actuator_joint_idxs)].set(
                    #         tau_g[jnp.array(self.task.actuator_joint_idxs)]
                    #     )
                    # )
                return mjx.step(model, x)
            
            # Advance the state for several steps, zero-order hold on control
            x = jax.lax.fori_loop(
                0,
                self.task.sim_steps_per_control_step,
                lambda _, x: _step_debug(_, x),
                x.replace(ctrl=u_mapped),
            )
            # jax.debug.print("After mjx.step")
            return x, (cost, sites)
        
        # jax.debug.print("NaN check — any NaNs in controls: {}", jnp.isnan(controls).any())
        # jax.debug.print("Inf check — any Infs in controls: {}", jnp.isinf(controls).any())
        # jax.debug.print("qpos shape: {}, dtype: {}, norm: {}", state.qpos.shape, state.qpos.dtype, jnp.linalg.norm(state.qpos))
        # jax.debug.print("qvel shape: {}, dtype: {}, norm: {}", state.qvel.shape, state.qvel.dtype, jnp.linalg.norm(state.qvel))

        # jax.debug.print("Starting rollout with controls: {}", controls)
        
        # Old 
        # final_state, (states, costs, trace_sites) = jax.lax.scan(
        #     _scan_fn, state, controls
        # )
        final_state, (costs, trace_sites) = jax.lax.scan(
            _scan_fn, state, controls
        )
        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        # Old
        # costs = jnp.append(costs, final_cost)
        # trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)
        
        costs = jnp.concatenate([costs, final_cost[None]], axis=0)
        trace_sites = jnp.concatenate([trace_sites, final_trace_sites[None]], axis=0)

        # Old
        # return states, Trajectory(
        #     controls=controls,
        #     costs=costs,
        #     trace_sites=trace_sites,
        # )
        return Trajectory(
            controls=None, # not needed, only creates duplication
            costs=costs,
            trace_sites=trace_sites,
        )

    @abstractmethod
    def init_params(self, seed: int = 0) -> Any:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Returns:
            The initial policy parameters.
        """
        pass

    @abstractmethod
    def sample_controls(self, params: Any) -> Tuple[jax.Array, Any]:
        """Sample a set of control sequences U ~ π(params).

        Args:
            params: Parameters of the policy distribution (e.g., mean, std).

        Returns:
            A control sequences U, size (num rollouts, horizon - 1).
            Updated parameters (e.g., with a new PRNG key).
        """
        pass

    @abstractmethod
    def update_params(self, params: Any, rollouts: Trajectory) -> Any:
        """Update the policy parameters π(params) using the rollouts.

        Args:
            params: The current policy parameters.
            rollouts: The rollouts obtained from the current policy.

        Returns:
            The updated policy parameters.
        """
        pass

    @abstractmethod
    def get_action(self, params: Any, t: float) -> jax.Array:
        """Get the control action at a given point along the trajectory.

        Args:
            params: The policy parameters, U ~ π(params).
            t: The time (in seconds) from the start of the trajectory.

        Returns:
            The control action u(t).
        """
        pass
