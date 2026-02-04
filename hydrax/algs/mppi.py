from typing import Tuple, Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass

# for state bins
#from hydrax.alg_base_visuals import SamplingBasedController, Trajectory
from hydrax.alg_base_opt import SamplingBasedController, Trajectory
from hydrax.algs.alg_extension_utils import colorize_time_series, make_savgol_filter, shift_tensor, savgol_coeffs



from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class MPPIParams:
    """Policy parameters for model-predictive path integral control.

    Attributes:
        mean: The mean of the control distribution, μ = [u₀, u₁, ..., ].
        rng: The pseudo-random number generator key.
    """
    rng: jax.Array
    spline: jax.Array


class MPPI(SamplingBasedController):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_randomizations: int = 1,
        alpha: float = 0.0,
        risk_strategy: RiskStrategy = None,
        colorize_noise: bool = False,   # !experimental
        alpha_noise: float = 3.0,  # !experimental
        shift: bool = False, # !experimental
        planning_freq: int = 1, # !experimental
        default_zero_controls: bool = False, #!experimental
        savgol_filter: bool = False, # !experimental
        seed: int = 0,
        update_cov: bool = True,  
    ):
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
        """
        super().__init__(task, num_randomizations, risk_strategy, seed)
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        self.alpha = alpha
        self.update_cov = update_cov
        
        # colored noise
        self.colorize_noise = colorize_noise
        self.alpha_noise = alpha_noise
        # shift
        self.shift = shift
        self.last_a_idx = int(self.task.dt * planning_freq)
        replan_period = 1 / planning_freq
        prediction_horizon = self.task.planning_horizon * self.task.dt
        self.last_a_idx = jnp.floor(replan_period / prediction_horizon * self.task.planning_horizon).astype(jnp.int32)
        print(f"MPPI last_a_idx: {self.last_a_idx}")
        
        self.default_zero_controls = default_zero_controls
        
        self.savgol_filter = savgol_filter  
        if savgol_filter:
            self.savgol_filter_fn = make_savgol_filter(window_length=7, polyorder=3, axis=1)

    def init_params(self, seed: int = 0, init_ctrl: jax.Array = None) -> MPPIParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        if init_ctrl is not None:
            # stack to full horizon
            spline = jnp.tile(init_ctrl, reps=(self.task.planning_horizon, 1))
        else:
            spline = jnp.zeros((self.task.planning_horizon, self.task.nu))
        return MPPIParams(spline=spline, rng=rng)

    def sample_controls(
        self, params: MPPIParams
    ) -> Tuple[jax.Array, MPPIParams]:
        """Sample a control sequence."""
        if self.shift:
            params = params.replace(
                spline=shift_tensor(params.spline, self.last_a_idx+1),
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
            noise = colorize_time_series(noise, remove_dc=False, alpha_noise=self.alpha_noise) # !experimental
        controls = params.spline + self.noise_level * noise
        
        # default zero controls
        if self.default_zero_controls:
            controls = controls.at[0, ...].set(jnp.zeros((self.task.planning_horizon, self.task.nu)))
        
        if self.savgol_filter:
            # controls are (samples x horizon x nu)
            controls = self.savgol_filter_fn(controls)
        # clip
        controls = jnp.clip(controls, self.task.u_min, self.task.u_max)
        # smoothen controls via 
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIParams, rollouts: Trajectory
    ) -> MPPIParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jnp.nan_to_num(jax.nn.softmax(-costs / self.temperature, axis=0))
        spline = jnp.sum(weights[:, None, None] * rollouts.controls, axis=0)
        spline = spline + self.alpha * (params.spline - spline)
        return params.replace(spline=spline)

    def get_action(self, params: MPPIParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt 
        idx = jnp.floor(idx_float).astype(jnp.int32)
        action = params.spline[idx]
        return action
    
