from typing import Tuple, Optional, Callable

from numpy import cov

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from functools import partial
from hydrax.alg_base_opt import SamplingBasedController, Trajectory

from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from .splines.akima import poly_akima, poly_interpolation
from .splines.bsplines import compute_b_spline_matrix
from hydrax.algs.alg_extension_utils import colorize_time_series, make_savgol_filter, shift_tensor, savgol_coeffs


@dataclass
class MTPParams:
    """Policy parameters for model-predictive path integral control.
    """
    rng: jax.Array
    mean: jax.Array = None
    cov: jax.Array = None
    spline: jax.Array = None
    elites: jax.Array = None   # (num_elites, T, U), optional
  


@partial(jax.jit, static_argnums=1)
def interpolate_path(path: jax.Array, num_points: int) -> jax.Array:
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y, n: jnp.linspace(x, y, n + 1)[:-1]
    return jax.vmap(linspace, in_axes=(0, 0, None))(start, goal, num_points).reshape(-1, path.shape[-1])


class MTP(SamplingBasedController):
    """Model Tensor Planning."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        M: int = 3,
        N: int = 50,
        degree: int = 2,
        num_elites: int = 5,
        sigma_start: float = 0.5,
        sigma_min: float = 0.1,
        sigma_max: float = 1.0,
        temperature: float = 0.1,
        num_randomizations: int = 1,
        beta: float = 0.1,
        alpha: float = 0.5,
        interpolation: str = 'akima',
        sample_weighting: str = 'cem-softmax',
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        update_cov: bool = True,
        colorize_noise: bool = False,   # !experimental
        alpha_noise: float = 3.0,  # !experimental
        shift: bool = False, # !experimental,
        planning_freq: int = 1, # !experimental
        keep_elites: int = 1,   #!experimental
        savgol_filter: bool = False, # !experimental
        default_zero_controls: bool = False, #!experimental
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
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_start = sigma_start
        control_dtype = jnp.float32#getattr(self.task.u_min, "dtype", jnp.float32)
        self.aknots = jnp.linspace(1, self.M+1, self.M+1, dtype=control_dtype)
        if keep_elites > num_elites:
            print(f"Warning: keep_elites ({keep_elites}) > num_elites ({num_elites}). Setting keep_elites = num_elites.")
            self.keep_elites = num_elites
        elif keep_elites < 1:
            print(f"Warning: keep_elites ({keep_elites}) < 1. Setting keep_elites = 1.")
            self.keep_elites = 1
        else:
            self.keep_elites = keep_elites
        self.bknots = self.start_clamped_knot_vector((self.M + 1), self.degree, dtype=control_dtype)
        self.bmat = jnp.asarray(
            compute_b_spline_matrix(
                self.bknots, self.degree, self.task.planning_horizon, dtype=control_dtype
            ),
            dtype=control_dtype,
        )
        self.temperature = temperature
        self.interpolation = interpolation
        self.alpha = alpha

        self.sample_weighting = sample_weighting
        self.control_mapper = task.make_control_mapper()
        
        self.nbr_mtp_samples = int(self.num_samples * self.beta)
        self.nbr_mppi_samples = self.num_samples - self.nbr_mtp_samples - 1
        
        self.colorize_noise = colorize_noise
        self.alpha_noise = alpha_noise
        # shift
        self.shift = shift
        self.last_a_idx = int(self.task.dt * planning_freq)
        
        self.default_zero_controls = default_zero_controls

        self.update_cov = update_cov

        self.savgol_filter = savgol_filter  
        if savgol_filter:
            self.savgol_filter_fn = make_savgol_filter(window_length=7, polyorder=2, axis=1)
        
        
    def start_clamped_knot_vector(self, num_ctrl_points, degree, dtype=jnp.float32):
        # First p+1 knots are the same
        n_knots = num_ctrl_points + degree + 1
        start = jnp.zeros(degree + 1, dtype=dtype)
        end = jnp.ones(degree + 1, dtype=dtype)
 
        # The rest increase uniformly
        nbr_rest = n_knots - 2 * (degree + 1) + 1
        rest = (jnp.arange(1, nbr_rest, dtype=dtype)) * (1.0 / jnp.array(nbr_rest, dtype=dtype)) 
        return jnp.concatenate([start, rest, end])

    def init_params(self, seed: int = 0) -> MTPParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        # spline = jnp.zeros((self.task.planning_horizon, self.task.nu))
        # sample mean with initial variance
        noise = jax.random.normal(
            rng,
            (
                self.task.planning_horizon,
                self.task.nu,
            ),
        )
        mean = jnp.zeros((self.task.planning_horizon, self.task.nu)) + self.sigma_start *noise
        spline = mean.copy()
        elites = spline[None, ...].repeat(self.keep_elites, axis=0)
        cov = jnp.full_like(mean, self.sigma_start)
        
        return MTPParams(rng=rng, 
                         spline=spline, 
                         mean=mean, 
                         elites=elites,
                         cov=cov,)

    
    def sample_controls(
        self, params: MTPParams
    ) -> Tuple[jax.Array, MTPParams]:
        
        """Sample a control sequence."""
        T = self.task.planning_horizon
        U = self.task.nu
        
        if self.shift:
            params = params.replace(
                mean=shift_tensor(params.mean, self.last_a_idx+1),
                spline=shift_tensor(params.spline, self.last_a_idx+1),
                elites=shift_tensor(params.elites, self.last_a_idx+1),
            )
        

        rng = params.rng
        # pre-allocate for memory efficiency
        out = jnp.empty((self.num_samples, T, U), dtype=jnp.float32)
        
        # The previous spline is included as a sample
        # TODO - can we incorporate elites here?
        # controls = params.spline[None, ...]
        out = out.at[0].set(params.spline)
        
        if self.nbr_mtp_samples > 1:
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
            layer_indices = jax.random.randint(sample_rng, (self.nbr_mtp_samples, self.M), 0, self.N - 1)
            def get_path(path_id: jax.Array) -> jax.Array:
                return control_points[jnp.arange(self.M), path_id]
            control_points = jax.vmap(get_path)(layer_indices)
          
            # add last_a_index of spline as first control point
            # !! Double Check
            init_sample = jnp.repeat(params.spline[self.last_a_idx][None, None, :], self.nbr_mtp_samples, axis=0)
            control_points = jnp.concatenate(
                [init_sample, control_points], axis=1
            )
            
            # interpolate the control points
            T = self.task.planning_horizon
            U = self.task.nu
            num_interp = T // (self.M )
            remain = T - num_interp * (self.M )      # tail we fill with the last control point
            interp_len = T - remain                      # number of samples produced by interpolation
            if self.interpolation == 'akima':
                # A: (B, M-1, 4, U)
                A = jax.vmap(poly_akima, in_axes=(None, 0))(self.aknots, control_points)
                # interp_vals: (B, interp_len, U)
                interp_vals = poly_interpolation(A, num_interp)

                # Preallocate full (B, T, U) and fill slices
                mtp_controls_full = jnp.empty((control_points.shape[0], T, U), dtype=interp_vals.dtype)
                mtp_controls_full = mtp_controls_full.at[:, :interp_len].set(interp_vals)

                last_cp = control_points[:, -1, :]                                 # (B, U)
                tail   = jnp.repeat(last_cp[:, None, :], remain, axis=1)           # (B, remain, U)
                mtp_controls_full = mtp_controls_full.at[:, interp_len:].set(tail) # (B, T, U)

            elif self.interpolation == 'bspline':
                # TODO this has to incorporate the last taken control as a first point + interpolate
                mtp_controls_full = jnp.einsum("...md,hm->...hd", control_points, self.bmat)
            elif self.interpolation == 'linear':
                # interp_vals: (B, interp_len, U)
                interp_vals = jax.vmap(interpolate_path, in_axes=(0, None))(control_points, num_interp)

                mtp_controls_full = jnp.empty((control_points.shape[0], T, U), dtype=interp_vals.dtype)
                mtp_controls_full = mtp_controls_full.at[:, :interp_len].set(interp_vals)

                last_cp = control_points[:, -1, :]                                 # (B, U)
                tail   = jnp.repeat(last_cp[:, None, :], remain, axis=1)           # (B, remain, U)
                mtp_controls_full = mtp_controls_full.at[:, interp_len:].set(tail) # (B, T, U)

            else:
                raise ValueError(f"Invalid sampling strategy: {self.interpolation}")
            # controls = jnp.concatenate([controls, mtp_controls], axis=0)
            out = out.at[1:1+self.nbr_mtp_samples].set(mtp_controls_full)

        if self.nbr_mppi_samples > 0:
            # Sample mppi_samples control sequences
            rng, sample_rng = jax.random.split(rng)
            noise = jax.random.normal(
                sample_rng,
                (
                    self.nbr_mppi_samples,
                    self.task.planning_horizon,
                    self.task.nu,
                ),
            )
            mppi_controls = params.mean + self.sigma_start * noise
            if self.savgol_filter:
                mppi_controls = self.savgol_filter_fn(mppi_controls)
            out = out.at[1+self.nbr_mtp_samples:1+self.nbr_mtp_samples+self.nbr_mppi_samples].set(mppi_controls)
        if self.keep_elites > 0 and params.elites is not None:
            out = out.at[:self.keep_elites].set(params.elites)
        # default zero controls
        if self.default_zero_controls:
            out = out.at[-1, ...].set(jnp.zeros((self.task.planning_horizon, self.task.nu)))
        # clip
        out = jnp.clip(out, self.task.u_min, self.task.u_max)
        return out, params.replace(rng=rng)

   
    def update_params(
        self, params: MTPParams, rollouts: Trajectory
    ) -> MTPParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps, MC
        
        if self.sample_weighting == 'cem-softmax':
            # CEM update with softmax weighting for the elites
            vals, elite_indices = jax.lax.top_k(-costs, self.num_elites)
            # elite_indices = jnp.argsort(costs)[:self.num_elites]
            controls = rollouts.controls[elite_indices]
            weights = jnp.nan_to_num(jax.nn.softmax(-costs[elite_indices] / self.temperature, axis=0))
            # The new proposal distribution is a Gaussian fit to the elites.
            weighted_controls = weights[:, None, None] * controls
            next_idx = elite_indices[0]  # use the best elite as control
        elif self.sample_weighting == 'cem':
            # CEM update with equal weighting for the elites
            vals, elite_indices = jax.lax.top_k(-costs, self.num_elites)
            # elite_indices = jnp.argsort(costs)[:self.num_elites]
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
        mean = mean + self.alpha * (params.mean - mean)
        
        if self.update_cov:
            cov = jnp.sqrt(jnp.sum(weights[:, None, None] * (controls - mean) ** 2, axis=0))
            cov = cov + self.alpha * (params.cov - cov)
            cov = jnp.clip(cov, self.sigma_min, self.sigma_max)
        else:
            cov = params.cov

        spline = rollouts.controls[next_idx]  # use the best elite as control
        new_elites = rollouts.controls[elite_indices[:self.keep_elites]]

        return params.replace(mean=mean, cov=cov, spline=spline, elites=new_elites)


    def get_action(self, params: MTPParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt 
        idx = jnp.floor(idx_float).astype(jnp.int32)
        action = params.spline[idx]
        return action
