from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base_opt import SamplingBasedController, Trajectory
# from alg_base_visuals import SamplingBasedController, Trajectory


from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from .splines.akima import poly_akima, poly_interpolation
from .splines.bsplines import compute_b_spline_matrix
from .splines.linear import interpolate_linear
from hydrax.algs.alg_extension_utils import colorize_time_series, shift_tensor

@dataclass
class AnMTPParams:
    rng: jax.Array
    mean: jax.Array = None     # (T, U)
    cov: jax.Array = None      # (T, U)
    spline: jax.Array = None   # (T, U)
    elites: jax.Array = None   # (num_elites, T, U), optional
    beta: jax.Array = None     # scalar jax array, updated inside jit


class AnMTP(SamplingBasedController):
    """Adaptive-beta Model Tensor Planning (updated to optimized MTP core).

    This version mirrors the sampling & interpolation fast-paths from MTP,
    but adapts the mixing proportion `beta` online from elite statistics.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        *,
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
        beta_lr: float = 0.2,        # adaptation step size
        beta_decay: float = 0.9,
        beta_min: float = 0.05,
        beta_max: float = 0.95,
        alpha: float = 0.5,
        interpolation: str = "akima",  # {"bspline","akima","linear"}
        sample_weighting: str = "cem-softmax",
        risk_strategy: RiskStrategy | None = None,
        colorize_noise: bool = False,   # !experimental
        shift: bool = False, # !experimental,
        planning_freq: int = 1, # !experimental
        keep_elites: int = 1,   #!experimental
        default_zero_controls: bool = False, #!experimental
        seed: int = 0,
        beta_strategy: str = "task", # task, ratio, greedy
        update_cov: bool = True,
    ):
        super().__init__(task, num_randomizations, risk_strategy, seed)
        assert degree >= 2, "degree must be at least 2."

        self.degree = degree
        self.N = N
        self.M = M
        self.beta = beta
        self.beta_lr = beta_lr
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_strategy = beta_strategy
        self.beta_decay = beta_decay

        self.num_samples = num_samples
        self.num_elites = num_elites
        if keep_elites > num_elites:
            print(f"Warning: keep_elites ({keep_elites}) > num_elites ({num_elites}). Setting keep_elites = num_elites.")
            self.keep_elites = num_elites
        elif keep_elites < 1:
            print(f"Warning: keep_elites ({keep_elites}) < 1. Setting keep_elites = 1.")
            self.keep_elites = 1
        else:
            self.keep_elites = keep_elites
        if sample_weighting not in ["cem", "cem-softmax", "mppi"]:
            raise ValueError(f"Invalid sample_weighting: {sample_weighting}")
        self.sample_weighting = sample_weighting
        if sample_weighting == "mppi":
            print("Info: sample_weighting='mppi' is not influenced by elites, all samples are weighted.")
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_start = sigma_start
        self.temperature = temperature
        self.interpolation = interpolation
        self.alpha = alpha
        self.update_cov = update_cov    

        control_dtype = jnp.float32
        self.aknots = jnp.linspace(1, self.M+1, self.M+1, dtype=control_dtype)
        # start-clamped knot vector for (M+1) control points (prepend current control)
        self.bknots = self._start_clamped_knot_vector((self.M + 1), self.degree, dtype=control_dtype)
        self.bmat = jnp.asarray(
            compute_b_spline_matrix(self.bknots, self.degree, self.task.planning_horizon, dtype=control_dtype),
            dtype=control_dtype,
        )  # (T, M+1)

        self.control_mapper = task.make_control_mapper()
        
        # ----------------------
        # Experimental
        self.colorize_noise = colorize_noise
        self.alpha_noise = 1.0  # 0=white, 1=pink, 2=brown
        # shift
        self.shift = shift
        self.last_a_idx = int(self.task.dt * planning_freq)
        
        self.default_zero_controls = default_zero_controls
        # ----------------------

    def _start_clamped_knot_vector(self, num_ctrl_points: int, degree: int, *, dtype=jnp.float32) -> jax.Array:
        n_knots = num_ctrl_points + degree + 1
        start = jnp.zeros(degree + 1, dtype=dtype)
        end = jnp.ones(degree + 1, dtype=dtype)
        # interior knots uniform in (0,1)
        nbr_rest = n_knots - 2 * (degree + 1) + 1
        rest = (jnp.arange(1, nbr_rest, dtype=dtype)) * (1.0 / jnp.array(nbr_rest, dtype=dtype))
        return jnp.concatenate([start, rest, end])

    # ----------------------
    # Parameters
    # ----------------------
    def init_params(self, seed: int = 0) -> AnMTPParams:
        rng = jax.random.key(seed)
        T, U = self.task.planning_horizon, self.task.nu
        
        noise = jax.random.normal(
            rng,
            (
                self.task.planning_horizon,
                self.task.nu,
            ),
        )
        mean = jnp.zeros((self.task.planning_horizon, self.task.nu)) + self.sigma_start *noise
        spline = mean.copy()
        # set all elites to spline
        elites = spline[None, ...].repeat(self.keep_elites, axis=0)

        cov = jnp.full_like(mean, self.sigma_start)
        beta = jnp.array(self.beta, dtype=jnp.float32)
        return AnMTPParams(rng=rng, spline=spline, mean=mean, cov=cov, beta=beta, elites=elites)

    # ----------------------
    # Sampling
    # ----------------------
    def sample_controls(self, params: AnMTPParams) -> Tuple[jax.Array, AnMTPParams]:
        rng = params.rng
        T, U = self.task.planning_horizon, self.task.nu
        R = self.num_samples
        
        # shift mean, spline, elites according to rolled out actions index
        if self.shift:
            params = params.replace(
                mean=shift_tensor(params.mean, self.last_a_idx+1),
                spline=shift_tensor(params.spline, self.last_a_idx+1),
                elites=shift_tensor(params.elites, self.last_a_idx+1),
                cov=shift_tensor(params.cov, self.last_a_idx+1) if self.update_cov else params.cov,
            )

        # Always fill the whole (R, T, U):
        # slot 0 = deterministic previous spline, slots 1..R-1 = sampled branch (masked MTP or MPPI)
        out = jnp.empty((R, T, U), dtype=jnp.float32)
        out = out.at[:self.keep_elites].set(params.elites)

        S = R - self.keep_elites  # number of stochastic samples with fixed shape

        # --- MTP branch (full S, later masked) ---
        rng, cp_key, pick_key = jax.random.split(rng, 3)
        # (M, N, U)
        cp_base = jax.random.uniform(cp_key, (self.M, self.N, U), minval=self.task.u_min, maxval=self.task.u_max)
        # (S, M) indices
        layer_idx = jax.random.randint(pick_key, (S, self.M), 0, self.N)
        # Gather one control point per layer using vmap to avoid shape/rank pitfalls
        def _gather(cp, idx_row):
            # cp: (M, N, U); idx_row: (M,)
            return cp[jnp.arange(self.M), idx_row, :]
        chosen = jax.vmap(_gather, in_axes=(None, 0))(cp_base, layer_idx)  # (S, M, U)
        init_cp = jnp.broadcast_to(params.spline[params.last_a_idx], (S, U))
        chosen = jnp.concatenate([init_cp[:, None, :], chosen], axis=1)  # (S, M+1, U)

        if self.interpolation == "bspline":
            mtp_controls = jnp.einsum("tm,bmu->btu", self.bmat, chosen)  # (S,T,U)
        elif self.interpolation == "akima":
            num_interp = T // (self.M)
            remain = T - num_interp * (self.M)
            A = jax.vmap(poly_akima, in_axes=(None, 0))(self.aknots, chosen)  # (S, M-1, 4, U)
            interp = poly_interpolation(A, num_interp)                        # (S, T - remain, U)
            mtp_controls = jnp.empty((S, T, U), dtype=interp.dtype)
            mtp_controls = mtp_controls.at[:, :T - remain].set(interp)
            mtp_controls = mtp_controls.at[:, T - remain :].set(jnp.repeat(chosen[:, -1:, :], remain, axis=1))
        elif self.interpolation == "linear":
            num_interp = T // (self.M )
            remain = T - num_interp * (self.M)
            interp = jax.vmap(interpolate_linear, in_axes=(0, None))(chosen, num_interp)
            mtp_controls = jnp.empty((S, T, U), dtype=interp.dtype)
            mtp_controls = mtp_controls.at[:, :T - remain].set(interp)
            mtp_controls = mtp_controls.at[:, T - remain :].set(jnp.repeat(chosen[:, -1:, :], remain, axis=1))
        else:
            raise ValueError(f"Invalid interpolation: {self.interpolation}")

        # --- MPPI branch (full S, later masked) ---
        rng, noise_key = jax.random.split(rng)
        noise = jax.random.normal(noise_key, (S, T, U))
        if self.colorize_noise:
            noise = self.colorize_time_series(noise) # !experimental
        mppi_controls = params.mean[None, ...] + params.cov[None, ...] * noise  # (S,T,U)

        # --- Masked mixing with fixed shape ---
        # K = floor(beta * S) determines how many of the S slots are MTP; shape is constant.
        K = jnp.floor(params.beta * S).astype(jnp.int32)
        mask = (jnp.arange(S) < K)[:, None, None]  # (S,1,1) boolean
        mixed_tail = jnp.where(mask, mtp_controls, mppi_controls)  # (S,T,U)

        out = out.at[self.keep_elites:].set(mixed_tail)
        # set elites
        if self.keep_elites > 0 and params.elites is not None:
            out = out.at[:self.keep_elites].set(params.elites)
        if self.default_zero_controls:
            out = out.at[-1, ...].set(jnp.zeros((self.task.planning_horizon, self.task.nu)))
        out = jnp.clip(out, self.task.u_min, self.task.u_max)
        return out, params.replace(rng=rng)

    # ----------------------
    # Parameter updates (+ adaptive beta)
    # ----------------------
    def update_params(self, params: AnMTPParams, rollouts: Trajectory) -> AnMTPParams:
        # Aggregate over time per sample
        costs = jnp.sum(rollouts.costs, axis=1)  # (R,)
        R = rollouts.controls.shape[0]
        S = R - 1

        # Elites + mean update (unchanged)
        if self.sample_weighting == "cem-softmax":
            vals, elite_idx = jax.lax.top_k(-costs, self.num_elites)
            controls = rollouts.controls[elite_idx]
            w = jnp.nan_to_num(jax.nn.softmax(-costs[elite_idx] / self.temperature, axis=0))
            weighted_controls = w[:, None, None] * controls
            next_idx = elite_idx[0]
        elif self.sample_weighting == "cem":
            vals, elite_idx = jax.lax.top_k(-costs, self.num_elites)
            controls = rollouts.controls[elite_idx]
            weighted_controls = controls / self.num_elites
            next_idx = elite_idx[0]
        elif self.sample_weighting == "mppi":
            controls = rollouts.controls
            w = jnp.nan_to_num(jax.nn.softmax(-costs / self.temperature, axis=0))
            weighted_controls = w[:, None, None] * controls
            next_idx = jnp.argmin(costs)
        else:
            raise ValueError(f"Unknown sample_weighting: {self.sample_weighting}")

        mean = jnp.sum(weighted_controls, axis=0)
        mean = mean + self.alpha * (params.mean - mean)
        # cov update
        if self.update_cov:
            cov = jnp.sqrt(jnp.sum(w[:, None, None] * (controls - mean) ** 2, axis=0))
            cov = cov + self.alpha * (params.cov - cov)
            cov = jnp.clip(cov, self.sigma_min, self.sigma_max)
        else:
            cov = params.cov
        spline = rollouts.controls[next_idx]

        # ----------------------
        # Beta updates
        # ----------------------
        K = jnp.floor(params.beta * S).astype(jnp.int32)
        # Tail indices for elites that are not the deterministic slot 0
        tail_idx = jnp.clip(elite_idx - 1, 0, S - 1)
        is_tail = elite_idx != 0
        is_mtp_slot = (tail_idx < K) # number of elites in MTP slots
        denom = jnp.maximum(1, is_tail.sum())  # avoid div-by-zero if all elites pick slot 0
        if self.beta_strategy == "task":
            new_beta = params.beta
        elif self.beta_strategy == "greedy":
            # increase beta if best elite is MTP, decrease otherwise
            new_beta = jnp.where(
                is_mtp_slot[0],
                self.beta_max,
                self.beta_decay * params.beta,
            )
            # set new mean to spline if best elite is from MTP
            # mean = jax.lax.cond(is_mtp_slot[0],
            #                 lambda _: spline,
            #                 lambda _: mean,
            #                 operand=None)

        elif self.beta_strategy == "ratio":
            # increase beta if fraction of MTP elites is above threshold, decrease otherwise
            frac_mtp_in_elites = (is_mtp_slot & is_tail).sum() / denom
            new_beta = jnp.where(
                frac_mtp_in_elites > 0.0,
                (1.0 - self.beta_lr) * params.beta + self.beta_lr * self.beta_max,
                self.beta_decay * params.beta,
            )
        new_beta = jnp.clip(new_beta, self.beta_min, self.beta_max)
        new_elites = rollouts.controls[elite_idx[:self.keep_elites]]

        return params.replace(mean=mean, spline=spline, beta=new_beta, new_elites=new_elites, cov=cov)

    # ----------------------
    # Action extraction
    # ----------------------
    def get_action(self, params: AnMTPParams, t: float) -> jax.Array:
        idx = jnp.floor(t / self.task.dt).astype(jnp.int32)
        return params.spline[idx]

    # Optional manual override
    def update_beta(self, beta: float, params: AnMTPParams) -> AnMTPParams:
        # self.beta = float(jnp.clip(beta, self.beta_min, self.beta_max))
        return params.replace(beta=beta)

