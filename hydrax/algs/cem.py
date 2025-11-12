from typing import Tuple, Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base_opt import SamplingBasedController, Trajectory
from hydrax.algs.alg_extension_utils import colorize_time_series, shift_tensor


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
    mean: jax.Array
    cov: jax.Array
    elites: jax.Array = None   # (num_elites, T, U), optional


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
        update_cov: bool = True,
        colorize_noise: bool = False,   # !experimental
        alpha_noise: float = 3.0,  # !experimental
        shift: bool = False, # !experimental,
        planning_freq: int = 1, # !experimental
        keep_elites: int = 1,   #!experimental
        default_zero_controls: bool = False, #!experimental
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
        self.last_a_idx = int(self.task.dt * planning_freq)
        
        if keep_elites > num_elites:
            print(f"Warning: keep_elites ({keep_elites}) > num_elites ({num_elites}). Setting keep_elites = num_elites.")
            self.keep_elites = num_elites
        elif keep_elites < 1:
            print(f"Warning: keep_elites ({keep_elites}) < 1. Setting keep_elites = 1.")
            self.keep_elites = 1
        else:
            self.keep_elites = keep_elites
        self.default_zero_controls = default_zero_controls

    def init_params(self, seed: int = 0) -> CEMParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        mean = jnp.zeros((self.task.planning_horizon, self.task.nu))
        cov = jnp.full_like(mean, self.sigma_start)
        elites = mean[None, ...].repeat(self.keep_elites, axis=0)
        return CEMParams(mean=mean, cov=cov, rng=rng, elites=elites)

    def sample_controls(self, params: CEMParams) -> Tuple[jax.Array, CEMParams]:
        """Sample a control sequence."""
        if self.shift:
            params = params.replace(
                mean=shift_tensor(params.mean, self.last_a_idx+1),
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
        controls = params.mean + params.cov * noise
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
        mean = jnp.mean(rollouts.controls[elites], axis=0)
        cov = params.cov
        # cov = jnp.maximum(
        #     jnp.std(rollouts.controls[elites], axis=0), self.sigma_min
        # )
        mean = mean + self.alpha * (params.mean - mean)
        if self.update_cov:
            cov = jnp.std(rollouts.controls[elites], axis=0)
            cov = jnp.clip(cov, a_min=self.sigma_min, a_max=self.sigma_max)
        new_elites = rollouts.controls[elites[:self.keep_elites]]
        return params.replace(mean=mean, cov=cov, elites=new_elites)

    def get_action(self, params: CEMParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt 
        idx = jnp.floor(idx_float).astype(jnp.int32)
        action = params.mean[idx]
        return action
