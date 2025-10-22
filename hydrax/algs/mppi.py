from typing import Tuple, Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass

# for state bins
#from hydrax.alg_base_visuals import SamplingBasedController, Trajectory
from hydrax.alg_base_opt import SamplingBasedController, Trajectory


from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class MPPIParams:
    """Policy parameters for model-predictive path integral control.

    Attributes:
        mean: The mean of the control distribution, μ = [u₀, u₁, ..., ].
        rng: The pseudo-random number generator key.
    """

    mean: jax.Array
    rng: jax.Array


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
        seed: int = 0,
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
        
        self.colorize_noise = colorize_noise
        self.alpha_noise = 3.0  # 0=white, 1=pink, 2=brown

    def init_params(self, seed: int = 0) -> MPPIParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        mean = jnp.zeros((self.task.planning_horizon, self.task.nu))
        return MPPIParams(mean=mean, rng=rng)

    def sample_controls(
        self, params: MPPIParams
    ) -> Tuple[jax.Array, MPPIParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.task.planning_horizon,
                self.task.nu,
            ),
        )
        if self.colorize_noise:
            noise = self.colorize_time_series(noise) # !experimental
        controls = params.mean + self.noise_level * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIParams, rollouts: Trajectory
    ) -> MPPIParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jnp.nan_to_num(jax.nn.softmax(-costs / self.temperature, axis=0))
        mean = jnp.sum(weights[:, None, None] * rollouts.controls, axis=0)
        mean = mean + self.alpha * (params.mean - mean)
        return params.replace(mean=mean)

    def get_action(self, params: MPPIParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        return params.mean[idx]
    
    # ----------------------
    # Experimental
    def colorize_time_series(self, noise, remove_dc=True, eps=1e-8):
        """
        noise: (B, T, D) white ~ N(0,1)
        Returns colored noise with approx unit variance per (B,D) trajectory.
        alpha=0 -> white, 1 -> pink (1/f), 2 -> brown (1/f^2)
        """
        B, T, D = noise.shape

        # reshape to (B*D, T) to FFT each series independently
        x = noise.reshape(B * D, T)

        # FFT -> apply magnitude shaping -> IFFT
        Xf = jnp.fft.rfft(x, axis=-1)                           # (B*D, T_r)
        freqs = jnp.fft.rfftfreq(T)                             # (T_r,)
        H = (1.0 / jnp.maximum(freqs, eps)) ** (self.alpha_noise / 2.0)    # magnitude shaping
        if remove_dc:
            H = H.at[0].set(0.0)
        Yf = Xf * H[None, :]                                    # broadcast
        y = jnp.fft.irfft(Yf, n=T, axis=-1)                     # (B*D, T)

        # de-mean and unit-std per series (robust for finite T)
        y = y - y.mean(axis=-1, keepdims=True)
        y = y / (y.std(axis=-1, keepdims=True) + eps)

        return y.reshape(B, T, D)
