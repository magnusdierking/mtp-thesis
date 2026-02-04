from typing import Tuple, Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base_opt import SamplingBasedController, Trajectory
from hydrax.algs.alg_extension_utils import colorize_time_series, make_savgol_filter, shift_tensor, savgol_coeffs


from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class CEMParams:
    """Policy parameters for the cross-entropy method.

    Attributes:
        mean: The mean of the control distribution, μ = [u₀, u₁, ..., ].
        cov: The (diagonal) covariance of the control distribution.
        rng: The pseudo-random number generator key.
    """
    rng: jax.Array
    spline: jax.Array
    cov: jax.Array
    elites: jax.Array = None   # (num_elites, T, U), optional
    predicted_state: jax.Array = None  # optional
    domain_weights: jax.Array = None  # optional


class CEM(SamplingBasedController):
    """Cross-entropy method with diagonal covariance."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        num_elites: int,
        sigma_start: float,
        sigma_min: float,
        sigma_max: float = 1.0,
        alpha: float = 0.5,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        update_cov: bool = False,
        colorize_noise: bool = False,   # !experimental
        alpha_noise: float = 3.0,  # !experimental
        shift: bool = False, # !experimental,
        planning_freq: int = 1, # !experimental
        keep_elites: int = 1,   #!experimental
        default_zero_controls: bool = False, #!experimental
        savgol_filter: bool = False, # !experimental
    ):
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            num_elites: The number of elite samples to keep at each iteration.
            sigma_start: The initial standard deviation for the controls.
            sigma_min: The minimum standard deviation for the controls.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
        """
        super().__init__(task, num_randomizations, risk_strategy, seed)
        self.num_samples = num_samples
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_start = sigma_start
        self.num_elites = num_elites
        self.alpha = alpha
        self.update_cov = update_cov
        
        self.colorize_noise = colorize_noise
        self.alpha_noise = alpha_noise
        # shift
        self.shift = shift
        replan_period = 1 / planning_freq
        prediction_horizon = self.task.planning_horizon * self.task.dt
        self.last_a_idx = int(replan_period / prediction_horizon * self.task.planning_horizon)
        print(f"CEM last_a_idx: {self.last_a_idx}")
        
        
        if keep_elites > num_elites:
            print(f"Warning: keep_elites ({keep_elites}) > num_elites ({num_elites}). Setting keep_elites = num_elites.")
            self.keep_elites = num_elites
        elif keep_elites < 1:
            print(f"Warning: keep_elites ({keep_elites}) < 1. Setting keep_elites = 1.")
            self.keep_elites = 1
        else:
            self.keep_elites = keep_elites
        self.default_zero_controls = default_zero_controls
        self.savgol_filter = savgol_filter  
        if savgol_filter:
            self.savgol_filter_fn = make_savgol_filter(window_length=7, polyorder=2, axis=1)

    def init_params(self, seed: int = 0) -> CEMParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        spline = jnp.zeros((self.task.planning_horizon, self.task.nu))
        cov = jnp.full_like(spline, self.sigma_start)
        elites = spline[None, ...].repeat(self.keep_elites, axis=0)
        predicted_state = jnp.zeros((self.num_randomizations, len(self.task.trace_site_ids), 7), dtype=jnp.float32) # ! experimental
        domain_weights = jnp.ones((self.num_randomizations,), dtype=jnp.float32) / self.num_randomizations # ! experimental
        
        return CEMParams(spline=spline, cov=cov, 
                         rng=rng, elites=elites, 
                         predicted_state=predicted_state,
                         domain_weights=domain_weights,
                         )

    def sample_controls(self, params: CEMParams) -> Tuple[jax.Array, CEMParams]:
        """Sample a control sequence."""
        if self.shift:
            params = params.replace(
                spline=shift_tensor(params.spline, self.last_a_idx+1),
                cov=shift_tensor(params.cov, self.last_a_idx+1) if self.update_cov else params.cov,
            )
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.task.planning_horizon,
                self.task.nu,
            ),
        )
        # colorize noise
        if self.colorize_noise:
            noise = self.colorize_time_series(noise, remove_dc=True, alpha=self.alpha_noise) # !experimental
        controls = params.spline + params.cov * noise
        if self.savgol_filter:
            controls = self.savgol_filter_fn(controls)
        # infuse elites from previous iteration
        if self.keep_elites > 0 and params.elites is not None:
            controls = controls.at[:self.keep_elites].set(params.elites)
        # default zero controls
        if self.default_zero_controls:
            controls = controls.at[self.keep_elites, ...].set(jnp.zeros((self.task.planning_horizon, self.task.nu)))
        # clipping
        controls = jnp.clip(controls, self.task.u_min, self.task.u_max)
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: CEMParams, rollouts: Trajectory
    ) -> CEMParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps

        # Sort the costs and get the indices of the elites.
        indices = jnp.argsort(costs)
        elites = indices[: self.num_elites]

        # The new proposal distribution is a Gaussian fit to the elites.
        spline = jnp.mean(rollouts.controls[elites], axis=0)
        cov = params.cov
        # cov = jnp.maximum(
        #     jnp.std(rollouts.controls[elites], axis=0), self.sigma_min
        # )
        spline = spline + self.alpha * (params.spline - spline)
        if self.update_cov:
            cov = jnp.std(rollouts.controls[elites], axis=0)
            cov = jnp.clip(cov, a_min=self.sigma_min, a_max=self.sigma_max)
        new_elites = rollouts.controls[elites[:self.keep_elites]]
        # rollouts.trace_sites is (domains, samples, steps, sites, 7)
        predicted_state = rollouts.trace_sites[:, indices[0], -1, ...] # one timestep over all domains, for rolloed out 
        
        return params.replace(spline=spline, cov=cov, elites=new_elites, predicted_state=predicted_state)

    def get_action(self, params: CEMParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt 
        idx = jnp.floor(idx_float).astype(jnp.int32)
        jax.debug.print("CEM get_action idx: {}", idx)
        action = params.spline[idx]
        return action
