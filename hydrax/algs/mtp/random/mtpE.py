from typing import Tuple

import jax
from jax import debug, make_jaxpr
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial
from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.algs.mtp.splines.akima import poly_akima, poly_interpolation
from hydrax.algs.mtp.splines.bsplines import compute_b_spline_matrix
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class MTPEParams:
    """Policy parameters for model-predictive path integral control.
    """
    rng: jax.Array
    mean: jax.Array = None
    cov: jax.Array = None
    spline: jax.Array = None
    vertices: jax.Array = None  # Variable control points for tensor sampling
    weights: jax.Array = None  # Weights for the vertices
    
    graph_indices: jax.Array = None  # Indices to spread the layers over the planning horizon


@partial(jax.jit, static_argnums=1)
def interpolate_path(path: jax.Array, num_points: int) -> jax.Array:
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y, n: jnp.linspace(x, y, n + 1)[:-1]
    return jax.vmap(linspace, in_axes=(0, 0, None))(start, goal, num_points).reshape(-1, path.shape[-1])


class MTPE(SamplingBasedController):
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
        self.bknots = jnp.arange(self.M + self.degree + 1)
        self.bmat = compute_b_spline_matrix(self.bknots, self.degree, self.task.planning_horizon)
        self.temperature = temperature
        self.interpolation = interpolation
        self.alpha = alpha

    def init_params(self, seed: int = 0) -> MTPEParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        spline = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        
        # M layers can be different from horizon, compute indices
        # to spread the layers over the planning horizon
        graph_indices = jnp.linspace(0, self.task.planning_horizon - 1, self.M).astype(jnp.int32)
        print("Graph Indices:", graph_indices)
        
        # if mu_init is not None:
        #     # repeat mean control to the planning horizon
        #     mean = jnp.repeat(mu_init[None, ...], self.task.planning_horizon)
        # else:
        #     # initialize mean control to zero
        mean = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        # TODO more sophisticated sigma
        cov = self.sigma_start * jnp.ones(self.M)
        # cov = jnp.full_like(mean, self.sigma_start)
        
        vertices = jax.random.uniform(
            rng,
            (self.M, self.N, self.task.model.nu),
            minval=self.task.u_min,
            maxval=self.task.u_max,
        )
        # uniform 1/ N weight for each layer
        weights = jnp.ones((self.M, self.N)) / self.N
        
        return MTPEParams(rng=rng, spline=spline, mean=mean, cov=cov, 
                          vertices=vertices, weights=weights, graph_indices=graph_indices)


    def _shift_graph(self, params: MTPEParams) -> MTPEParams:
        # M Layers, N samples per layer, nu control dimension
        # shift the graph by one layer
        rng, sample_rng = jax.random.split(params.rng)
        # sample new layer for end of graph (uniformly distributed)
        # TODO - does it make sense to sample accroding to the second to last layer ?
        control_points = jax.random.uniform(
            sample_rng,
            (self.N, self.task.model.nu),
            minval=self.task.u_min,
            maxval=self.task.u_max,
        )
        # shift the graph by one layer
        new_vertices = jnp.roll(params.vertices, -1, axis=0)
        new_vertices = new_vertices.at[-1].set(control_points)
        # update the weights for the new layer
        new_weights = jnp.roll(params.weights, -1, axis=0)
        # uniform 1/ N weight for the new layer
        new_weights = params.weights.at[-1].set(jnp.ones((self.N,)) / self.N)
        # update the random number generator
        params.replace(rng=rng, 
                       vertices=new_vertices, 
                       weights=new_weights)
        return params




    def sample_controls(
        self, params: MTPEParams
    ) -> Tuple[jax.Array, MTPEParams]:
        """Sample a control sequence."""
        rng = params.rng
        mtp_samples = int(self.num_samples * self.beta)
        # The previous spline is included as a sample
        controls = params.spline[None, ...]
        if mtp_samples > 1:


            # TODO - can we reduce the number of control points without sacrificing too muhc ?
            # sample M control points per layer
            control_points = params.vertices.transpose(1, 0, 2)  # (N, nu, M)
            
            # interpolate the control points
            if self.interpolation == 'akima':
                A = jax.vmap(poly_akima, in_axes=(None, 0))(self.aknots, control_points) # (batch, M - 1, 4, nu)
                num_interp = self.task.planning_horizon // (self.M - 1)
                remain = self.task.planning_horizon - num_interp * (self.M - 1)
                mtp_controls = poly_interpolation(A, num_interp)
                remain_controls = jnp.repeat(control_points[:, -1, None], remain, axis=1)
                mtp_controls = jnp.concatenate([mtp_controls, remain_controls], axis=1)
            elif self.interpolation == 'bspline':
                mtp_controls = jnp.einsum("...md,hm->...hd", control_points, self.bmat)

            elif self.interpolation == 'linear':
                num_interp = self.task.planning_horizon // (self.M - 1)
                remain = self.task.planning_horizon - num_interp * (self.M - 1)
                mtp_controls = jax.vmap(interpolate_path, in_axes=(0, None))(control_points, num_interp)
                remain_controls = jnp.repeat(control_points[:, -1, None], remain, axis=1)
                mtp_controls = jnp.concatenate([mtp_controls, remain_controls], axis=1)
            else:
                raise ValueError(f"Invalid sampling strategy: {self.interpolation}")
            controls = jnp.concatenate([controls, mtp_controls], axis=0)

        mppi_samples = self.num_samples - mtp_samples - 1
        if mppi_samples > 0:
            # Sample mppi_samples control sequences
            rng, sample_rng = jax.random.split(rng)
            noise = jax.random.normal(
                sample_rng,
                (
                    mppi_samples,
                    self.task.planning_horizon,
                    self.task.model.nu,
                ),
            )
            mppi_controls = params.mean + params.cov * noise
            controls = jnp.concatenate([controls, mppi_controls], axis=0)

        return controls, params.replace(rng=rng)
    

    def _effective_sample_size(self, weights):
        """Compute the effective sample size for a 1D array."""
        return 1.0 / jnp.sum(weights ** 2)



    def _resample_layer(self, weights, rng, threshold):
        """Resample the weights if ESS < threshold."""
        ess = self._effective_sample_size(weights)

        def do_resample(_):
            top_indices = jnp.argsort(weights)[-weights.shape[0] // 2:]
            resampled = jax.random.choice(rng, top_indices, shape=(weights.shape[0],), replace=True)
            return resampled

        def skip_resample(_):
            return jnp.arange(weights.shape[0])

        return jax.lax.cond(ess < threshold, do_resample, skip_resample, operand=None)
    
    
    def _apply_resampling(self, vertices, resampled_indices):
        """
        samples: jnp.ndarray of shape (M, N, d)
        resampled_indices: jnp.ndarray of shape (M, N)
        Returns: resampled_samples of shape (M, N, d)
        """
        def resample_single_layer(layer_samples, layer_indices):
            return jnp.take(layer_samples, layer_indices, axis=0)

        return jax.vmap(resample_single_layer, in_axes=(0, 0))(vertices, resampled_indices)




    def update_params(
        self, params: MTPEParams, rollouts: Trajectory
    ) -> MTPEParams:
        """Update the mean with an exponentially weighted average."""
        
        rng = params.rng
        print("Rollouts:", rollouts.controls.shape)
        
        # TODO - compute accumulated reward
        cost = jnp.cumsum(rollouts.costs, axis=1)[1:, params.graph_indices]  # cumulative cost over time steps, shape (num_samples, planning_horizon)
        cost = jnp.swapaxes(cost, 0, 1)  # swap axes to have (planning_horizon, num_samples)
        
        costs = jnp.sum(cost, axis=1)  # sum over time steps,
        print("Costs:", costs.shape)
        print("Costs:", cost.shape)
        
        # TODO - reweight the samples according to the costs
        weights=jax.nn.softmax(-cost / self.temperature, axis=0) # softmax over samples, (nbr_samples, layers + 1)
        print("Weights:", weights.shape) 
        
        # TODO - check for degeneracy of the samples, resample if necessary
        resample_fn = jax.vmap(self._resample_layer, in_axes=(0, None, None)) 
        
        resampled_indices = resample_fn(weights, rng, 0.5)  # TODO threshold
        print("Resampled Indices:", resampled_indices.shape)
        print("Vertices:", params.vertices.shape)
        # resample the vertices according to the resampled indices
        vertices = self._apply_resampling(params.vertices, resampled_indices)  # (M, N, nu)
        print("Vertices 2:", vertices.shape)
        
        # TODO exrtact best option for control
        elite_indices = jnp.argsort(costs)[0] # index of the best sample
        print("Elite Costs:", costs)
        print("Elite Indices:", elite_indices)
        best_spline = rollouts.controls[elite_indices]  # shape (planning_horizon, nu)
                
        # TODO diffuse
        vertices = vertices + params.cov[:, None, None] * jax.random.normal(key=rng, shape=vertices.shape)  # diffuse the vertices with noise

        return params.replace(weights=weights,
                              vertices=vertices,
                              rng=rng,
                              spline=best_spline)


    def get_action(self, params: MTPEParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        action = params.spline[idx]
        print("Action:",action)
        # TODO we shift here because we dont want shift if controller is faster than the task
        self._shift_graph(params)
        return action
