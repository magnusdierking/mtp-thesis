from typing import Tuple, Optional, Callable

from numpy import cov

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from functools import partial
from hydrax.alg_base_opt import SamplingBasedController, Trajectory
# from hydrax.alg_base_visuals import SamplingBasedController, Trajectory

from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from .splines.akima import poly_akima, poly_interpolation
from .splines.bsplines import compute_b_spline_matrix


@dataclass
class MTPParams:
    """Policy parameters for model-predictive path integral control.
    """
    rng: jax.Array
    mean: jax.Array = None
    cov: jax.Array = None
    spline: jax.Array = None
    last_a_idx: int = 0


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
        self.aknots = jnp.linspace(1, self.M, self.M)
        # self.bknots = jnp.arange(self.M + self.degree + 1)
        # B spline matrix including the current spline as control point
        self.bknots = self.start_clamped_knot_vector((self.M + 1), self.degree)
        self.bmat = compute_b_spline_matrix(self.bknots, self.degree, self.task.planning_horizon)
        self.temperature = temperature
        self.interpolation = interpolation
        self.alpha = alpha

        self.sample_weighting = sample_weighting
        self.control_mapper = task.make_control_mapper()
        
        self.nbr_mtp_samples = int(self.num_samples * self.beta)
        self.nbr_mppi_samples = self.num_samples - self.nbr_mtp_samples - 1
        
        
    def start_clamped_knot_vector(self, num_ctrl_points, degree):
        # First p+1 knots are the same
        n_knots = num_ctrl_points + degree + 1
        start = jnp.zeros(degree + 1, dtype=float)
        end = jnp.ones(degree + 1, dtype=float) 
        # The rest increase uniformly
        nbr_rest = n_knots - 2 * (degree + 1) + 1
        rest = (jnp.arange(1,nbr_rest, dtype=float)) * (1.0 / (nbr_rest))  # Normalize to [0, 1]
        return jnp.concatenate([start, rest, end])

    def init_params(self, seed: int = 0) -> MTPParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        spline = jnp.zeros((self.task.planning_horizon, self.task.nu))
        mean = jnp.zeros((self.task.planning_horizon, self.task.nu))
        cov = jnp.full_like(mean, self.sigma_start)
        
        return MTPParams(rng=rng, 
                         spline=spline, 
                         mean=mean, 
                         cov=cov)

    
    def sample_controls(
        self, params: MTPParams
    ) -> Tuple[jax.Array, MTPParams]:
        
        """Sample a control sequence."""
        T = self.task.planning_horizon
        U = self.task.nu

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
            init_sample = jnp.repeat(params.spline[params.last_a_idx][None, None, :], self.nbr_mtp_samples, axis=0)
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
            # controls = jnp.concatenate([controls, mtp_controls], axis=0)
            out = out.at[1:1+self.nbr_mtp_samples].set(mtp_controls)

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
            mppi_controls = params.mean + params.cov * noise
            out = out.at[1+self.nbr_mtp_samples:1+self.nbr_mtp_samples+self.nbr_mppi_samples].set(mppi_controls)
            # controls = jnp.concatenate([controls, mppi_controls], axis=0)

        return out, params.replace(rng=rng)
    
   
    def update_params(
        self, params: MTPParams, rollouts: Trajectory
    ) -> MTPParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps, MC
        
        if self.sample_weighting == 'cem-softmax':
            # CEM update with softmax weighting for the elites
            vals, elite_indices = jax.lax.top_k(-costs, self.num_elites)
            elite_indices = elite_indices  # indices of smallest costs
            # elite_indices = jnp.argsort(costs)[:self.num_elites]
            controls = rollouts.controls[elite_indices]
            weights = jnp.nan_to_num(jax.nn.softmax(-costs[elite_indices] / self.temperature, axis=0))
            # The new proposal distribution is a Gaussian fit to the elites.
            weighted_controls = weights[:, None, None] * controls
            next_idx = elite_indices[0]  # use the best elite as control
        elif self.sample_weighting == 'cem':
            # CEM update with equal weighting for the elites
            vals, elite_indices = jax.lax.top_k(-costs, self.num_elites)
            elite_indices = elite_indices  # indices of smallest costs
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
        # TODO - can we change this ?
        #cov = jnp.sqrt(jnp.sum(weights[:, None, None] * (controls - mean) ** 2, axis=0))
        #cov = jnp.clip(cov, self.sigma_min, self.sigma_max)
        mean = mean + self.alpha * (params.mean - mean)
        #cov = cov + self.alpha * (params.cov - cov)
        spline = rollouts.controls[next_idx]  # use the best elite as control

        # return params.replace(mean=mean, cov=cov, spline=spline)
        return params.replace(mean=mean, spline=spline)


    def get_action(self, params: MTPParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        params = params.replace(last_a_idx=idx)
        action = params.spline[idx]
        return action
