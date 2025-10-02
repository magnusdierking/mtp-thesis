from typing import Tuple, Optional, Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from functools import partial
from hydrax.alg_base import SamplingBasedController, Trajectory
# from hydrax.alg_base_visuals import SamplingBasedController, Trajectory

from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from ..splines.akima import poly_akima, poly_interpolation
from ..splines.bsplines import compute_b_spline_matrix


@dataclass
class AnMTPParams:
    """Policy parameters for model-predictive path integral control.
    """
    rng: jax.Array
    mean: jax.Array = None         # (T, nu)
    cov: jax.Array = None          # (T, nu)
    spline: jax.Array = None       # (T, nu)
    last_elites: jax.Array = None  # (num_elites, T, nu)
    last_a_idx: int = None


@partial(jax.jit, static_argnums=1)
def interpolate_path(path: jax.Array, num_points: int) -> jax.Array:
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y, n: jnp.linspace(x, y, n + 1)[:-1]
    return jax.vmap(linspace, in_axes=(0, 0, None))(start, goal, num_points).reshape(-1, path.shape[-1])


class AnMTP(SamplingBasedController):
    """Model Tensor Planning."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        M: int = 3,
        N: int = 50,
        degree: int = 2,
        num_elites: int = 5,
        outer_iters: int = 1, # iterations of parameter updates per planning
        sigma_start: float = 0.5,
        sigma_min: float = 0.1,
        sigma_max: float = 1.0,
        temperature: float = 0.1,
        num_randomizations: int = 1,
        beta: float = 0.1,
        alpha: float = 0.5,
        kappa2: float = 0.35,
        interpolation: str = 'akima',
        sample_weighting: str = 'cem-softmax',
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        # control_mapper: Optional[Callable[[mjx.Model, mjx.Data, jax.Array], jax.Array]] = None,
    ):
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
        """
        super().__init__(task, num_randomizations, risk_strategy, seed)
        assert degree >= 2, "degree must be at least 2."
        self.degree = degree
        self.N = N
        self.M = M
        self.beta = beta
        self.kappa2 = kappa2
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.outer_iters = outer_iters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_start = sigma_start
        self.aknots = jnp.linspace(1, self.M, self.M)
        self.bknots = self.start_clamped_knot_vector((self.M + 1), self.degree) # + current point as sample
        self.bmat = compute_b_spline_matrix(self.bknots, self.degree, self.task.planning_horizon)
        self.temperature = temperature
        self.interpolation = interpolation
        self.alpha = alpha
        
        self.sample_weighting = sample_weighting
        self.control_mapper = task.make_control_mapper()
        
        # EXPERIMENTAL - noise coloring
        self.colorize_noise = True
        self.noise_alpha = 8.0  # higher is smoother
        
        
    def start_clamped_knot_vector(self, num_ctrl_points, degree):
        # First p+1 knots are the same
        n_knots = num_ctrl_points + degree + 1
        start = jnp.zeros(degree + 1, dtype=float)
        end = jnp.ones(degree + 1, dtype=float) 
        # The rest increase uniformly
        nbr_rest = n_knots - 2 * (degree + 1) + 1
        rest = (jnp.arange(1,nbr_rest, dtype=float)) * (1.0 / (nbr_rest))  # Normalize to [0, 1]
        return jnp.concatenate([start, rest, end])
        
    def init_params(self, seed: int = 0) -> AnMTPParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        spline = jnp.zeros((self.task.planning_horizon, self.task.nu))
        mean = jnp.zeros((self.task.planning_horizon, self.task.nu))
        cov = jnp.ones_like(mean) * jnp.array([self.act_annealed_variance(t) for t in range(self.task.planning_horizon)])[...,None]
        # cov = jnp.full_like(mean, self.sigma_start)
        return AnMTPParams(rng=rng, spline=spline, mean=mean, cov=cov, last_a_idx=0)
    
    def act_annealed_variance(self, i):
        return jnp.exp(-(self.task.planning_horizon - i) * self.task.nu / (self.kappa2 * self.task.planning_horizon))

    def update_beta(self, beta: float):
        self.beta = beta

    # TODO 
    # def traj_annealed_variance(self, i):
    #     return jnp.exp(-(self.task.planning_horizon - i) * self.task.nu / (self.kappa2 * self.task.planning_horizon))
    
    
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
        H = (1.0 / jnp.maximum(freqs, eps)) ** (self.alpha / 2.0)    # magnitude shaping
        if remove_dc:
            H = H.at[0].set(0.0)
        Yf = Xf * H[None, :]                                    # broadcast
        y = jnp.fft.irfft(Yf, n=T, axis=-1)                     # (B*D, T)

        # de-mean and unit-std per series (robust for finite T)
        y = y - y.mean(axis=-1, keepdims=True)
        y = y / (y.std(axis=-1, keepdims=True) + eps)

        return y.reshape(B, T, D)
    

    def sample_controls(
        self, params: AnMTPParams
    ) -> Tuple[jax.Array, AnMTPParams]:
        """Sample a control sequence."""
        rng = params.rng
        
        # -----------------------------------------
        # roll all arrays accroding to which actions we already took
        # -----------------------------------------
        
        mean = jnp.roll(params.mean, shift=-params.last_a_idx, axis=0)
        cov = jnp.roll(params.cov, shift=-params.last_a_idx, axis=0)
        spline = jnp.roll(params.spline, shift=-params.last_a_idx, axis=0)

        last_idx = self.task.planning_horizon - params.last_a_idx
        mask = jnp.arange(self.task.planning_horizon) < last_idx
        
        mean = jnp.where(mask[..., None], mean, jnp.ones_like(mean) * jnp.take(mean, last_idx-1, 0))
        cov = jnp.where(mask[..., None], cov, jnp.ones_like(cov) * jnp.take(cov, last_idx-1, 0))
        spline = jnp.where(mask[..., None], spline, jnp.ones_like(spline) * jnp.take(spline, last_idx-1, 0))

        # The previous spline is included as a sample
        # TODO - include fixed number of prior elites here
        controls = params.spline[None, ...]
        
        #------------------------
        
        # Sample the control points for the MTP
        rng, sample_rng = jax.random.split(rng)
        control_points = jax.random.uniform(
            sample_rng,
            (
                self.M,
                self.N,
                self.task.nu,
            ),
            minval=self.task.u_min,
            maxval=self.task.u_max,
        )
        # sample points from the graph
        rng, sample_rng = jax.random.split(rng)
        layer_indices = jax.random.randint(rng, (self.num_samples - 1, self.M), 0, self.N - 1)
        
        def get_path(path_id: jax.Array) -> jax.Array:
            return control_points[jnp.arange(self.M), path_id]
        control_points = jax.vmap(get_path)(layer_indices)
        
        # add last_a_index of spline as first control point
        # b spline matrix is designed for one additional control point to account for this
        # !! Double Check
        init_sample = jnp.repeat(params.spline[params.last_a_idx][None, None, :], self.num_samples - 1, axis=0)
        control_points = jnp.concatenate(
            [init_sample, control_points], axis=1
        )
        
        # interpolate the control points
        if self.interpolation == 'akima':
            A = jax.vmap(poly_akima, in_axes=(None, 0))(self.aknots, control_points) # (batch, M - 1, 4, nu)
            num_interp = self.task.planning_horizon // (self.M - 1)
            remain = self.task.planning_horizon - num_interp * (self.M - 1)
            mtp_controls = poly_interpolation(A, num_interp)
            remain_controls = jnp.repeat(control_points[:, -1, None], remain, axis=1)
            mtp_controls = jnp.concatenate([mtp_controls, remain_controls], axis=1)
        elif self.interpolation == 'bspline':
            # TODO this has to incorporate the last taken control as a first point + interpolate
            mtp_controls = jnp.einsum("bmd,hm->bhd", control_points, self.bmat)
        elif self.interpolation == 'linear':
            num_interp = self.task.planning_horizon // (self.M - 1)
            remain = self.task.planning_horizon - num_interp * (self.M - 1)
            mtp_controls = jax.vmap(interpolate_path, in_axes=(0, None))(control_points, num_interp)
            remain_controls = jnp.repeat(control_points[:, -1, None], remain, axis=1)
            mtp_controls = jnp.concatenate([mtp_controls, remain_controls], axis=1)
        else:
            raise ValueError(f"Invalid sampling strategy: {self.interpolation}")
        controls = jnp.concatenate([controls, mtp_controls], axis=0)

    
        # Sample mppi_samples control sequences
        rng, sample_rng = jax.random.split(rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples - 1,
                self.task.planning_horizon,
                self.task.nu,
            ),
        )
        if self.colorize_noise:
            noise = self.colorize_time_series(noise)
        mppi_controls = params.mean + params.cov * noise

        # -----------------------------------------
        # Combine sampling strategies via masking
        # -----------------------------------------
        total_samples = self.num_samples - 1  # Python int set in __init__

        rng, idx_rng = jax.random.split(rng)

        # 1) Single permutation of a static length
        perm = jax.random.permutation(idx_rng, total_samples, independent=True)

        # 2) Rank array so we can threshold by a (possibly tracer) mtp_samples
        order = jnp.empty((total_samples,), dtype=jnp.int32).at[perm].set(
            jnp.arange(total_samples, dtype=jnp.int32)
        )

        # 3) Compute mtp_samples on device (no dynamic shapes!)
        mtp_samples = jnp.clip(
            jnp.floor(self.num_samples * self.beta).astype(jnp.int32),
            0, total_samples
        )

        # 4) Masks: first mtp_samples (in permutation order) are MTP; rest are MPPI
        mtp_mask  = order < mtp_samples
        mppi_mask = ~mtp_mask  # complementary within total_samples

        # 5) Apply masks (controls[:total_samples] excludes the reserved "last spline")
        mtp_selected  = mtp_controls  * mtp_mask[:, None, None]
        mppi_selected = mppi_controls * mppi_mask[:, None, None]
        final_result  = mtp_selected + mppi_selected

        controls = jnp.concatenate([controls, final_result], axis=0)

        return controls, params.replace(rng=rng, 
                                        mean=mean, 
                                        cov=cov, 
                                        spline=spline)

    def update_params(
        self, params: AnMTPParams, rollouts: Trajectory
    ) -> AnMTPParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        
        if self.sample_weighting == 'cem-softmax':
            # CEM update with softmax weighting for the elites
            elite_indices = jnp.argsort(costs)[:self.num_elites]
            controls = rollouts.controls[elite_indices]
            weights = jnp.nan_to_num(jax.nn.softmax(-costs[elite_indices] / self.temperature, axis=0))
            # The new proposal distribution is a Gaussian fit to the elites.
            weighted_controls = weights[:, None, None] * controls
            next_idx = elite_indices[0]  # use the best elite as control
        elif self.sample_weighting == 'cem':
            # CEM update with equal weighting for the elites
            elite_indices = jnp.argsort(costs)[:self.num_elites]
            controls = rollouts.controls[elite_indices]
            weighted_controls = controls / self.num_elites
            next_idx = elite_indices[0]
        elif self.sample_weighting == 'mppi':
            # Use all samples, weighted by the softmax of costs
            controls = rollouts.controls
            weights = jnp.nan_to_num(jax.nn.softmax(-costs / self.temperature, axis=0))
            weighted_controls = weights[:, None, None] * controls
            next_idx = jnp.argmin(costs)
        
        mean = jnp.sum(weighted_controls, axis=0)
        # TODO - can we change this ?
        # cov = jnp.sqrt(jnp.sum(weights[:, None, None] * (controls - mean) ** 2, axis=0))
        # cov = jnp.clip(cov, self.sigma_min, self.sigma_max)
        mean = mean + self.alpha * (params.mean - mean)
        # cov = cov + self.alpha * (params.cov - cov)
        spline = rollouts.controls[next_idx]  # use the best elite as control
        params = params.replace(mean=mean)

        return params.replace(spline=spline)

    def get_action(self, params: AnMTPParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        params = params.replace(last_a_idx=idx + 1)
        action = params.spline[idx]
        return action
