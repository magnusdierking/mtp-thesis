from typing import Tuple

import jax
from jax import debug, make_jaxpr
import jax.numpy as jnp
from flax.struct import dataclass
from functools import partial
from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from mtp.splines.akima import poly_akima, poly_interpolation
from mtp.splines.bsplines import compute_b_spline_matrix


@dataclass
class MTPThompsonParams:
    rng: jax.Array
    vertices: jax.Array = None       # Shape: (M, N) , graph layers by samples
    mean: jax.Array = None           # Shape: (M, N) , expected costs for each sample
    cov: jax.Array = None            # Shape: (M, N) , trust for each samples' expected costs # TODO multivariate ?
    counts: jax.Array = None         # Shape: (M, N) , number of times each sample was selected, used for variance scaling
    spline: jax.Array = None         # Shape: (planning_horizon, nu)
    graph_indices: jax.Array = None  # Indices in horizon where there are graph layers


@partial(jax.jit, static_argnums=1)
def interpolate_path(path: jax.Array, num_points: int) -> jax.Array:
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y, n: jnp.linspace(x, y, n + 1)[:-1]
    return jax.vmap(linspace, in_axes=(0, 0, None))(start, goal, num_points).reshape(-1, path.shape[-1])


class MTPThompson(SamplingBasedController):
    """Model Tensor Planning."""

    def __init__(
            self,
            task: Task,
            num_samples: int,
            k: int,
            M: int = 3,
            N: int = 50,
            degree: int = 2,
            sigma_min: float = 0.1,
            sigma_max: float = 1.0,
            temperature: float = 0.1,
            num_randomizations: int = 1,
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
        self.k = k
        self.num_samples = num_samples
        control_dtype = getattr(self.task.u_min, "dtype", jnp.float32)
        self.sigmas = jnp.linspace(sigma_min, sigma_max, M, dtype=control_dtype)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.aknots = jnp.linspace(1, self.M, self.M, dtype=control_dtype)
        self.bknots = jnp.arange(self.M + self.degree + 1, dtype=control_dtype)
        self.bmat = jnp.asarray(
            compute_b_spline_matrix(
                self.bknots, self.degree, self.task.planning_horizon, dtype=control_dtype
            ),
            dtype=control_dtype,
        )
        self.temperature = temperature
        self.interpolation = interpolation
        self.alpha = alpha

    def init_params(self, seed: int = 0) -> MTPThompsonParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        spline = jnp.zeros((self.task.planning_horizon, self.task.model.nu))

        # M layers can be different from horizon, compute indices
        # to spread the layers over the planning horizon
        graph_indices = jnp.linspace(0, self.task.planning_horizon - 1, self.M).astype(jnp.int32)
        print("Graph Indices:", graph_indices)

        vertices = jax.random.uniform(
            rng,
            (self.M, self.N, self.task.model.nu),
            minval=self.task.u_min,
            maxval=self.task.u_max,
        )
        # initialize thompson parameters, initially positive since costs are positive
        mean = jnp.ones((self.M, self.N))

        # each sample has its own sigma, so we need to expand the shape
        cov = jnp.repeat(self.sigmas[:, None], self.N, axis=1)  # (M, N)

        counts = jnp.zeros((self.M, self.N))  # counts for each sample, used for variance scaling

        return MTPThompsonParams(rng=rng,
                                 spline=spline,
                                 mean=mean,
                                 cov=cov,
                                 vertices=vertices,
                                 counts=counts,
                                 graph_indices=graph_indices)

    def _shift_graph(self, params: MTPThompsonParams) -> MTPThompsonParams:
        # M Layers, N samples per layer, nu control dimension
        # shift the graph by one layer
        rng, sample_rng = jax.random.split(params.rng)
        # sample new layer for end of graph (uniformly distributed)
        # TODO - does it make sense to sample informed by second to last layer ?
        control_points = jax.random.uniform(
            sample_rng,
            (self.N, self.task.model.nu),
            minval=self.task.u_min,
            maxval=self.task.u_max,
        )
        # shift the graph by one layer
        new_vertices = jnp.roll(params.vertices, -1, axis=0)
        new_vertices = new_vertices.at[-1].set(control_points)

        # shift the means
        new_means = jnp.roll(params.mean, -1, axis=0)
        new_means = new_means.at[-1].set(jnp.ones((self.N,)))

        # shift the covariances
        new_cov = jnp.roll(params.cov, -1, axis=0)
        new_cov = new_cov.at[-1].set(jnp.ones((self.N,)) * self.sigmas[-1])
        # shift the counts
        new_counts = jnp.roll(params.counts, -1, axis=0)
        new_counts = new_counts.at[-1].set(jnp.zeros((self.N,)))

        return params.replace(
            rng=rng,
            vertices=new_vertices,
            mean=new_means,
            cov=new_cov,
            counts=new_counts,
            spline=params.spline,
            graph_indices=params.graph_indices
        )

    def sample_controls(
            self, params: MTPThompsonParams
    ) -> Tuple[jax.Array, MTPThompsonParams]:
        """Sample a control sequence."""
        rng = params.rng

        # The previous spline is included as a sample
        controls = params.spline[None, ...]

        # Thompson sampling,
        samples = jax.random.normal(rng, (self.M, self.N))  # (M, N)
        samples = params.mean + samples * params.cov[:, None]

        # normalize samples per layer
        weights = jax.nn.softmax(samples / self.temperature, axis=1)  # (M, N)

        # for each layer, pick the best k samples according to the weights
        top_indices = jnp.argsort(weights, axis=1)[:, -self.k:]

        # trajectories are all possible control sequences of the top k samples
        def get_path(path_id: jax.Array) -> jax.Array:
            return params.vertices[jnp.arange(self.M), path_id]
        # get the control points for the top k samples
        control_points = jax.vmap(get_path)(top_indices)

        print("Control points shape:", control_points.shape) # should be k^m
        # interpolate the control points
        if self.interpolation == 'akima':
            A = jax.vmap(poly_akima, in_axes=(None, 0))(self.aknots, control_points)  # (batch, M - 1, 4, nu)
            num_interp = self.task.planning_horizon // (self.M - 1)
            remain = self.task.planning_horizon - num_interp * (self.M - 1)
            mtp_controls = poly_interpolation(A, num_interp)
            remain_controls = jnp.repeat(control_points[:, -1, None], remain, axis=1)
            mtp_controls = jnp.concatenate([mtp_controls, remain_controls], axis=1)
        elif self.interpolation == 'bspline':
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

        return controls, params.replace(rng=rng,
                                        vertices=params.vertices,
                                        mean=params.mean,
                                        cov=params.cov,
                                        counts=params.counts,
                                        spline=mtp_controls,
                                        graph_indices=params.graph_indices)




    def update_params(
            self, params: MTPThompsonParams, rollouts: Trajectory
    ) -> MTPThompsonParams:

        # PROBLEM after first iteration, thompson sampling will be incredibly biased if we initialize the
        # expeected cost badly



        rng = params.rng
        print("Rollouts:", rollouts.controls.shape)

        # TODO - compute accumulated reward
        cost = jnp.cumsum(rollouts.costs, axis=1)[1:,
               params.graph_indices]  # cumulative cost over time steps, shape (num_samples, planning_horizon)
        cost = jnp.swapaxes(cost, 0, 1)  # swap axes to have (planning_horizon, num_samples)

        costs = jnp.sum(cost, axis=1)  # sum over time steps,
        print("Costs:", costs.shape)
        print("Costs:", cost.shape)

        # TODO - reweight the samples according to the costs
        weights = jax.nn.softmax(-cost / self.temperature, axis=0)  # softmax over samples, (nbr_samples, layers + 1)
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
        elite_indices = jnp.argsort(costs)[0]  # index of the best sample
        print("Elite Costs:", costs)
        print("Elite Indices:", elite_indices)
        best_spline = rollouts.controls[elite_indices]  # shape (planning_horizon, nu)

        # TODO diffuse
        vertices = vertices + params.cov[:, None, None] * jax.random.normal(key=rng,
                                                                            shape=vertices.shape)  # diffuse the vertices with noise

        return params.replace(weights=weights,
                              vertices=vertices,
                              rng=rng,
                              spline=best_spline)

    def get_action(self, params: MTPEParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        action = params.spline[idx]
        print("Action:", action)
        # TODO we shift here because we dont want shift if controller is faster than the task
        self._shift_graph(params)
        return action
