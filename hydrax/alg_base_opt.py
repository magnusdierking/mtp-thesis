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
    """Data container for MPC rollouts.

    Attributes:
        controls: Array of shape ``(R, T, U)`` where ``R`` is the number of
            sampled trajectories, ``T`` the planning horizon, and ``U`` the
            control dimension.
        costs: Array of shape ``(R, T + 1)`` holding accumulated running costs
            for each timestep plus the terminal cost.
        trace_sites: Array of shape ``(R, T + 1, S, 3)`` capturing the xyz
            coordinates of ``S`` traced sites for visualisation/debugging.
    """

    controls: jax.Array
    costs: jax.Array
    trace_sites: jax.Array

    def __len__(self):
        """Return the number of time steps in the trajectory (T)."""
        return self.costs.shape[-1] - 1


class SamplingBasedController(ABC):
    """Base class for sampling-based MPC algorithms.

    Subclasses implement the distribution over control trajectories and the
    associated parameter updates. All computations are expected to be JAX/JIT
    friendly.
    """

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
        
        # invariants
        self._T      = self.task.planning_horizon
        self._act_idx   = jnp.array(self.task.actuator_joint_idxs)
        self._dt        = jnp.asarray(self.task.dt, jnp.float32)
        self._qfrc0 = jnp.zeros((self.model.nv,), dtype=jnp.float32) # maybe not needed
        self._xfrc0 = jnp.zeros((self.model.nbody, 6), dtype=jnp.float32) # maybe not needed
        
        
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

    @jax.named_call
    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Advance the controller by one planning iteration.

        Args:
            state: Initial simulator state ``mjx.Data`` representing ``x₀``.
            params: Parameter pytree that defines the sampling distribution
                over control sequences.

        Returns:
            Tuple ``(params', trajectory)`` where ``params'`` is the updated
            parameter pytree and ``trajectory`` bundles the sampled rollouts.
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
    
            
    
    @jax.named_call
    def rollout_with_randomizations(self, state: mjx.Data, controls: jax.Array, rng: jax.Array):
        """Roll out control sequences across optional domain randomisations.

        Args:
            state: Base ``mjx.Data`` state ``x₀``.
            controls: Array of shape ``(R, T, U)`` containing the clipped
                control trajectories for each rollout.
            rng: PRNG key used when multiple domain randomisations are active.

        Returns:
            ``Trajectory`` with costs aggregated across randomisations via the
            configured ``RiskStrategy``.
        """

        # Tile the state to [NR, ...]
        def tile_state(s, n): return jax.tree.map(lambda a: jnp.broadcast_to(a, (n,) + a.shape), s)
        states = tile_state(state, self.num_randomizations)

        # Optionally randomize states per domain (NR keys)
        if self.num_randomizations > 1:
            keys = jax.random.split(rng, self.num_randomizations)
            rand = jax.vmap(self.task.domain_randomize_data)(states, keys)
            states = states.tree_replace(rand)

        # Run rollouts per domain: randomized_axes marks which model leaves have axis=0
        costs_NR_R_T, sites_NR_R_T = jax.vmap(self.run_one_randomization,
            in_axes=(self.randomized_axes, 0, None)
        )(self.model, states, controls)
        # shapes: costs_NR_R_T [NR, R, T], sites_NR_R_T [NR, R, T, ...]

        # Combine risk across domains for each rollout independently
        # Expecting combine_costs: [NR, T] -> [T]
        costs_R_T = jax.vmap(self.risk_strategy.combine_costs, in_axes=1)(
            costs_NR_R_T  # vmap over R (axis=1)
        )  # -> [R, T]

        # Controls are identical across domains; choose any domain for trace sites, or combine if desired
        trace_sites_R_T = sites_NR_R_T[0]  # [R, T, ...]

        return Trajectory(controls=controls, costs=costs_R_T, trace_sites=trace_sites_R_T)

        
        
    def rollout_with_randomizations2(self, state: mjx.Data, controls: jax.Array, rng: jax.Array):
        """
        controls: [R, T, U]
        returns:  Trajectory with costs [R, T+1] and trace_sites [R, T+1, ...]
        """
        # --- make [NR, ...] states ---
        def tile_state(s, n): return jax.tree.map(lambda a: jnp.broadcast_to(a, (n,) + a.shape), s)
        states_NR = tile_state(state, self.num_randomizations)

        # optional per-domain state randomization (you already do this)
        if self.num_randomizations > 1:
            keys = jax.random.split(rng, self.num_randomizations)
            rand = jax.vmap(self.task.domain_randomize_data)(states_NR, keys)
            states_NR = states_NR.tree_replace(rand)

        NR, R = self.num_randomizations, controls.shape[0]

        # --- flatten to one big batch B = NR*R ---
        # states: repeat each domain state R times  -> [B, ...]
        states_flat = jax.tree.map(lambda a: jnp.repeat(a, R, axis=0), states_NR)

        # controls: tile the R-block NR times       -> [B, T, U]
        controls_flat = jnp.tile(controls[None, ...], (NR, 1, 1, 1)).reshape(NR * R, *controls.shape[1:])

        # model: if randomized_axes is not None, those leaves are [NR, *] — repeat along 0 by R
        model_in = self.model
        in_axes_model = None
        if self.randomized_axes is not None:
            model_in = jax.tree.map(
                lambda leaf, ax: (jnp.repeat(leaf, R, axis=0) if ax == 0 else leaf),
                self.model, self.randomized_axes
            )
            # after repeating, model_in’s randomized leaves are [B, ...]
            in_axes_model = jax.tree.map(lambda ax: 0 if ax == 0 else None, self.randomized_axes)

        # --- one vmap over B ---
        costs_B, sites_B = jax.vmap(
            self.eval_rollout,
            in_axes=(in_axes_model, 0, 0) if in_axes_model is not None else (None, 0, 0)
        )(model_in, states_flat, controls_flat)   # costs_B: [B, T+1], sites_B: [B, T+1, ...]

        # fold back to [NR, R, ...]
        costs_NR_R = costs_B.reshape(NR, R, -1)
        sites_NR_R = jax.tree.map(lambda x: x.reshape(NR, R, *x.shape[1:]), sites_B)

        # combine risk across domains per rollout (your logic)
        costs_R_T1 = jax.vmap(self.risk_strategy.combine_costs, in_axes=1)(costs_NR_R)  # [R, T+1]

        # choose which trace to keep (or aggregate) — unchanged
        trace_sites_R_T1 = jax.tree.map(lambda x: x[0], sites_NR_R)  # pick domain 0

        return Trajectory(controls=controls, costs=costs_R_T1, trace_sites=trace_sites_R_T1)

        
        
    @jax.named_call
    def run_one_randomization(self, model_r: mjx.Model, state_r: mjx.Data, controls_all: jax.Array):
        """Roll out all sampled trajectories for a single domain realisation.

        Args:
            model_r: Randomised ``mjx.Model`` instance.
            state_r: Randomised ``mjx.Data`` state.
            controls_all: Array ``(R, T, U)`` of shared control sequences.

        Returns:
            Tuple ``(costs, trace_sites)`` where ``costs`` has shape ``(R, T + 1)``
            and ``trace_sites`` has shape ``(R, T + 1, S, 3)``.
        """
        with jax.named_scope("rollout_step"):
            costs_R, sites_R = jax.vmap(self.eval_rollout, in_axes=(None, None, 0))(
                model_r, state_r, controls_all
            )
        # shapes: costs_R [R, T], sites_R [R, T, ...]
        return costs_R, sites_R


    
    def eval_rollout(self, model: mjx.Model, state: mjx.Data, controls: jax.Array):
        """Execute one rollout for a fixed model and initial state.

        Args:
            model: mjx.Model describing the dynamics.
            state: mjx.Data state used as starting point.
            controls: Array (T, U) of control inputs for the horizon.

        Returns:
            (costs, trace_sites) with shapes (T + 1,) and (T + 1, S, 3).
        """
        # ---------- Hoisted constants & bound callables ----------
        dt = self.task.dt
        sim_steps = int(self.task.sim_steps_per_control_step)
        use_gc = self.gravity_compensator is not None
        act_idx = getattr(self, "_act_idx", None)

        running_cost = self.task.running_cost
        terminal_cost = self.task.terminal_cost
        get_sites = self.task.get_trace_sites
        control_mapper = self.control_mapper
        
        x0 = mjx.forward(model, state)  

        # ---------- One "macro" step: measure -> cost -> integrate ----------
        @jax.named_call
        def _macro_step(x: mjx.Data, u: jax.Array):

            u_mapped = control_mapper(x, u) if control_mapper is not None else u
            cost_t = dt * running_cost(x, u_mapped)
            sites_t   = jax.lax.stop_gradient(get_sites(x))
            # sites_t = get_sites(x_for_cost)

            # 4) Integrate sim_steps with constant control (u_mapped).
            if use_gc:
                tau_g = self.gravity_compensator(x)
                qfrc_t = self._qfrc0.at[act_idx].set(tau_g[act_idx])
                xfrc_t = self._xfrc0
                def _micro_step(i, x_cur):
                    x_pre = x_cur.replace(qfrc_applied=qfrc_t, xfrc_applied=xfrc_t)
                    return mjx.step(model, x_pre)
            else:
                def _micro_step(i, x_cur):
                    x_pre = x_cur.replace(qfrc_applied=self._qfrc0, xfrc_applied=self._xfrc0)
                    return mjx.step(model, x_pre)

            x_next = jax.lax.fori_loop(0, sim_steps, _micro_step, x.replace(ctrl=u_mapped))
            return x_next, (cost_t, sites_t)

        # ---------- Scan over control horizon ----------
        x_final, (costs, sites) = jax.lax.scan(_macro_step, x0, controls)

        # ---------- Terminal terms ----------
        term_cost = terminal_cost(x_final)
        term_sites= jax.lax.stop_gradient(get_sites(x_final))


        costs = jnp.empty((self._T+1,), costs.dtype).at[:-1].set(costs).at[-1].set(term_cost)
        trace_sites = jnp.empty((self._T+1,)+sites.shape[1:], sites.dtype) \
                    .at[:-1].set(sites).at[-1].set(term_sites)
        return costs, trace_sites

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


