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

        # ---------- One "macro" step: measure -> cost -> integrate ----------
        @jax.named_call
        def _macro_step(x: mjx.Data, u: jax.Array):
            # 1) Bring sensors/derived fields up-to-date for cost/obs.
            x_for_cost = mjx.forward(model, x)

            # 2) Map control if needed (keep this outside the micro loop).
            u_mapped = control_mapper(x_for_cost, u) if control_mapper is not None else u

            # 3) Running cost & trace sites at the *start* of this control interval.
            #    (matches your original semantics)
            cost_t = dt * running_cost(x_for_cost, u_mapped)
            sites_t   = jax.lax.stop_gradient(get_sites(x_for_cost))
            # sites_t = get_sites(x_for_cost)

            # 4) Integrate sim_steps with constant control (u_mapped).
            if use_gc:
                tau_g = self.gravity_compensator(x_for_cost)
                qfrc_t = self._qfrc0.at[act_idx].set(tau_g[act_idx])
                xfrc_t = self._xfrc0
                def _micro_step(i, x_cur):
                    x_pre = x_cur.replace(qfrc_applied=qfrc_t, xfrc_applied=xfrc_t)
                    return mjx.step(model, x_pre)
            else:
                def _micro_step(i, x_cur):
                    x_pre = x_cur.replace(qfrc_applied=self._qfrc0, xfrc_applied=self._xfrc0)
                    return mjx.step(model, x_pre)
            # def _micro_step(i, x_cur: mjx.Data):
            #     # Optionally add gravity compensation (device-side, no Python branch)
            #     def _with_gc(x_in):
            #         tau_g = self.gravity_compensator(x_in)
            #         qfrc = qfrc0.at[act_idx].set(tau_g[act_idx])
            #         return x_in.replace(qfrc_applied=qfrc, xfrc_applied=xfrc0)

            #     def _no_gc(x_in):
            #         return x_in.replace(qfrc_applied=qfrc0, xfrc_applied=xfrc0)

            #     x_pre = jax.lax.cond(use_gc, _with_gc, _no_gc, x_cur)
            #     # Step once with fixed control for this sub-step.
            #     return mjx.step(model, x_pre)

            x_next = jax.lax.fori_loop(0, sim_steps, _micro_step, x_for_cost.replace(ctrl=u_mapped))
            return x_next, (cost_t, sites_t)

        # ---------- Scan over control horizon ----------
        x_final, (costs, sites) = jax.lax.scan(_macro_step, state, controls)

        # ---------- Terminal terms ----------
        term_cost = terminal_cost(x_final)#[None]
        term_sites= jax.lax.stop_gradient(get_sites(x_final))#[None]
        # term_sites = get_sites(x_final)[None]

        costs = jnp.empty((self._T+1,), costs.dtype).at[:-1].set(costs).at[-1].set(term_cost)
        trace_sites = jnp.empty((self._T+1,)+sites.shape[1:], sites.dtype) \
                    .at[:-1].set(sites).at[-1].set(term_sites)

        # costs = jnp.concatenate([costs, term_cost], axis=0)       # [T+1]
        # trace_sites = jnp.concatenate([sites, term_sites], axis=0)  # [T+1, ...]
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


  # @jax.named_call
    # def eval_rollout(self, model: mjx.Model, state: mjx.Data, controls: jax.Array):
    #     """Execute one rollout for a fixed model and initial state.

    #     Args:
    #         model: ``mjx.Model`` describing the dynamics.
    #         state: ``mjx.Data`` state used as starting point.
    #         controls: Array ``(T, U)`` of control inputs for the horizon.

    #     Returns:
    #         Tuple ``(costs, trace_sites)`` with shapes ``(T + 1,)`` and
    #         ``(T + 1, S, 3)`` respectively.
    #     """
    #     def step_fn(x, u):
    #         # x = mjx.forward(model, x)
    #         u_mapped = self.control_mapper(x, u) if self.control_mapper else u
    #         cost = self.task.dt * self.task.running_cost(x, u_mapped)
    #         sites = self.task.get_trace_sites(x)

    #         def _micro(_, x):
    #             if self.gravity_compensator:
    #                 tau_g = self.gravity_compensator(x)
    #                 # qfrc = jnp.zeros_like(x.qfrc_applied).at[self._act_idx].set(
    #                 #     tau_g[self._act_idx]
    #                 # )
    #                 qfrc = self._qfrc0.at[self._act_idx].set(tau_g)
    #                 xfrc = self._xfrc0
    #                 x = x.replace(qfrc_applied=qfrc, 
    #                               xfrc_applied=xfrc)
    #             return mjx.step(model, x)

    #         x = jax.lax.fori_loop(0, self.task.sim_steps_per_control_step,
    #                             _micro, x.replace(ctrl=u_mapped))
    #         return x, (cost, sites)

    #     final_state, (costs, trace_sites) = jax.lax.scan(step_fn, state, controls)
    #     # terminal pieces
    #     final_cost = self.task.terminal_cost(final_state)[None]
    #     final_sites = self.task.get_trace_sites(final_state)[None]
    #     costs = jnp.concatenate([costs, final_cost], axis=0)          # [T]
    #     trace_sites = jnp.concatenate([trace_sites, final_sites], 0)  # [T, ...]
    #     return costs, trace_sites

    # @jax.named_call
    # def eval_rollout(self, model: mjx.Model, state: mjx.Data, controls: jax.Array):
    #     """Execute one rollout for a fixed model and initial state.

    #     Args:
    #         model: ``mjx.Model`` describing the dynamics.
    #         state: ``mjx.Data`` state used as starting point.
    #         controls: Array ``(T, U)`` of control inputs for the horizon.

    #     Returns:
    #         Tuple ``(costs, trace_sites)`` with shapes ``(T + 1,)`` and
    #         ``(T + 1, S, 3)`` respectively.
    #     """
    #     def step_fn(x, u):
    #         x = mjx.forward(model, x)
    #         u_mapped = self.control_mapper(x, u) if self.control_mapper else u

    #         # compute once per control step
    #         if self.gravity_compensator:
    #             tau_g = self.gravity_compensator(x)
    #             qfrc_t = self._qfrc0.at[self._act_idx].set(tau_g[self._act_idx])
    #             xfrc_t = self._xfrc0

    #             def _micro(_, x1):
    #                 x1 = x1.replace(qfrc_applied=qfrc_t, xfrc_applied=xfrc_t)
    #                 return mjx.step(model, x1)
    #         else:
    #             def _micro(_, x1):
    #                 return mjx.step(model, x1)

    #         # cost + sites (compute once for this u)
    #         cost = self._dt * self.task.running_cost(x, u_mapped)
    #         sites = jax.lax.stop_gradient(self.task.get_trace_sites(x))

    #         x = jax.lax.fori_loop(0, self.task.sim_steps_per_control_step,
    #                             _micro, x.replace(ctrl=u_mapped))
    #         return x, (cost, sites)

    #     final_state, (costs, trace_sites) = jax.lax.scan(step_fn, state, controls)
    #     final_cost  = self.task.terminal_cost(final_state)
    #     final_sites = self.task.get_trace_sites(final_state)

    #     T = controls.shape[0]
    #     costs_T1 = jnp.empty((T+1,), costs.dtype).at[:-1].set(costs).at[-1].set(final_cost)
    #     trace_T1 = jnp.empty((T+1,)+trace_sites.shape[1:], trace_sites.dtype).at[:-1].set(trace_sites).at[-1].set(final_sites)

    #     return costs_T1, trace_T1