from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from alg_base_opt_dr import SamplingBasedController, Trajectory
from hydrax.risk import RiskStrategy, ExpectedCost
from hydrax.task_base import Task
from hydrax.algs.mtp.splines.akima import poly_akima, poly_interpolation
from hydrax.algs.mtp.splines.bsplines import compute_b_spline_matrix
from hydrax.algs.mtp.splines.linear import interpolate_linear


@dataclass
class AnMTPParams:
    rng: jax.Array
    mean: jax.Array = None     # (T, U)
    cov: jax.Array = None      # (T, U)
    spline: jax.Array = None   # (T, U)
    elites: jax.Array = None   # (num_elites, T, U), optional
    # TODO for now this is simply the predicted state of the sites
    predicted_state: jax.Array = None  # (domains, sites, state), optional
    # TODO 
    domain_weights: jax.Array = None  # (domains,), optional 
    beta: jax.Array = None     # scalar jax array, updated inside jit
    last_a_idx: int = 0


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
        keep_elites: int = 1,  
        sigma_start: float = 0.5,
        sigma_min: float = 0.1,
        sigma_max: float = 1.0,
        temperature: float = 0.1,
        num_randomizations: int = 1,
        planning_frequency: int = 25,
        beta: float = 0.1,
        beta_lr: float = 0.2,        # adaptation step size
        beta_min: float = 0.0,
        beta_max: float = 0.95,
        alpha: float = 0.5,
        interpolation: str = "akima",  # {"bspline","akima","linear"}
        sample_weighting: str = "cem-softmax",
        risk_strategy: RiskStrategy | None = None,
        colorize_noise: bool = False,   # !experimental
        seed: int = 0,
        update_cov: bool = True,
    ):
        # super().__init__(task, num_randomizations, risk_strategy, seed)
        # ! experimental
        super().__init__(task, num_randomizations, ExpectedCost(jnp.ones((num_randomizations,), dtype=jnp.float32) / num_randomizations), seed)
        assert degree >= 2, "degree must be at least 2."

        self.degree = degree
        self.N = N
        self.M = M
        self.beta = beta
        self.beta_lr = beta_lr
        self.beta_min = beta_min
        self.beta_max = beta_max

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

        control_dtype = jnp.float32
        self.aknots = jnp.linspace(1, self.M, self.M, dtype=control_dtype)
        # start-clamped knot vector for (M+1) control points (prepend current control)
        self.bknots = self._start_clamped_knot_vector((self.M + 1), self.degree, dtype=control_dtype)
        self.bmat = jnp.asarray(
            compute_b_spline_matrix(self.bknots, self.degree, self.task.planning_horizon, dtype=control_dtype),
            dtype=control_dtype,
        )  # (T, M+1)

        self.control_mapper = task.make_control_mapper()
        
        # ----------------------
        # Experimental colored noise
        self.colorize_noise = colorize_noise
        self.alpha_noise = 1.0  # 0=white, 1=pink, 2=brown
        # ----------------------
        # Experimental domain randomization
        # compute horizon step to save and compare to next observed state
        horizon_time = self.task.planning_horizon * self.task.sim_steps_per_control_step * self.task.dt 
        planning_time = 1.0 / planning_frequency
        self.observation_step = jnp.round(horizon_time / planning_time) * self.task.planning_horizon 
        self.observation_step = self.observation_step.astype(jnp.int32)
        
        self.domain_weights = jnp.ones((self.num_randomizations,), dtype=jnp.float32) / self.num_randomizations
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
        spline = jnp.zeros((T, U), dtype=jnp.float32)
        elites = jnp.zeros((self.keep_elites, T, U), dtype=jnp.float32) # ! experimental
        predicted_state = jnp.zeros((self.num_randomizations, len(self.task.trace_site_ids), 3), dtype=jnp.float32) # ! experimental
        # domain_weights = jnp.ones((self.num_randomizations,), dtype=jnp.float32) / self.num_randomizations # ! experimental
        mean = jnp.zeros((T, U), dtype=jnp.float32)
        cov = jnp.full_like(mean, self.sigma_start)
        beta = jnp.array(self.beta, dtype=jnp.float32)
        return AnMTPParams(rng=rng, 
                           spline=spline, 
                           mean=mean, 
                           cov=cov, 
                           beta=beta, 
                           elites=elites, 
                           predicted_state=predicted_state)

    # ----------------------
    # Sampling
    # ----------------------
    def sample_controls(self, params: AnMTPParams) -> Tuple[jax.Array, AnMTPParams]:
        rng = params.rng
        T, U = self.task.planning_horizon, self.task.nu
        R = self.num_samples

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
        out = jnp.clip(out, self.task.u_min, self.task.u_max)
        return out, params.replace(rng=rng)

    # ----------------------
    # Parameter updates (+ adaptive beta)
    # ----------------------
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
        spline = rollouts.controls[next_idx]

        # --- adaptive beta from elites using mask logic with fixed shape ---
        K = jnp.floor(params.beta * S).astype(jnp.int32)
        # Tail indices for elites that are not the deterministic slot 0
        tail_idx = jnp.clip(elite_idx - 1, 0, S - 1)
        is_tail = elite_idx != 0
        is_mtp_slot = (tail_idx < K)
        # Consider only elites in tail; ignore deterministic slot when computing fraction
        denom = jnp.maximum(1, is_tail.sum())  # avoid div-by-zero if all elites pick slot 0
        frac_mtp_in_elites = (is_mtp_slot & is_tail).sum() / denom

        # increase beta if any elites are MTP, decrease otherwise
        # use jax operations to stay in-jit
        new_beta = jnp.where(
            frac_mtp_in_elites > params.beta,
            (1.0 - self.beta_lr) * params.beta + self.beta_lr * self.beta_max,
            (1.0 - self.beta_lr) * params.beta + self.beta_lr * self.beta_min,
        )
        # beta_target = frac_mtp_in_elites
        # new_beta = (1.0 - self.beta_lr) * params.beta + self.beta_lr * beta_target
        new_beta = jnp.clip(new_beta, self.beta_min, self.beta_max)
        
        # ! Experimental
        predicted_state = rollouts.trace_sites[:, next_idx, self.observation_step, ...] # one timestep over all domains, for rolloed out 

        return params.replace(mean=mean, spline=spline, beta=new_beta, elites=controls[:self.keep_elites], predicted_state=predicted_state)

    # ----------------------
    # Action extraction
    # ----------------------
    def get_action(self, params: AnMTPParams, t: float) -> jax.Array:
        idx = jnp.floor(t / self.task.dt).astype(jnp.int32)
        # store index so the next sampling step can prepend the last applied action
        params = params.replace(last_a_idx=idx)
        return params.spline[idx]

    # Optional manual override
    def update_beta(self, beta: float):
        self.beta = float(jnp.clip(beta, self.beta_min, self.beta_max))
        
        
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

    # def update_domain_randomization_model(self, rng: jax.Array, weights: jax.Array) -> None:
    #     rng, subrng = jax.random.split(rng)
    #     # check ESS
    #     ess = 1.0 / jnp.sum(jnp.square(weights))
    #     print("ESS:", ess)
    #     if ess < 0.9 * self.num_randomizations:
    #         # resample particles by sampling with replacement according to weights
    #         indices = jax.random.choice(
    #             subrng,
    #             self.num_randomizations,
    #             shape=(self.num_randomizations,),
    #             p=weights / jnp.sum(weights),
    #         )
    #         new_masses = self.model.body_mass[indices, self.task.T_bid]
    #         # perturb masses a bit
    #         rng, subrng = jax.random.split(rng)
    #         noise = jax.random.normal(subrng, (self.num_randomizations,)) * 0.05
    #         new_masses = jnp.clip(new_masses + noise, 0.1, 2.0)
    #         self.model = self.model.body_mass.at[:, self.task.T_bid].set(new_masses)
    #     else:
    #         new_masses = self.model.body_mass[:, self.task.T_bid]
    #         self.model = self.model.body_mass.at[:, self.task.T_bid].set(new_masses)
    def update_domain_randomization_model(self, rng: jax.Array, weights: jax.Array) -> bool:
        # normalize weights for ESS and sampling
        w = weights / (jnp.sum(weights) + 1e-12)
        ess = 1.0 / jnp.sum(w * w)
        # print("ESS:", ess)

        bid = self.task.T_bid
        B = self.num_randomizations

        rng, subrng = jax.random.split(rng)

        if ess < 0.9 * B:
            # resample indices (with replacement) on the batch axis
            idx = jax.random.choice(subrng, B, shape=(B,), p=w).astype(jnp.int32)

            # gather current masses for that body across the chosen instances
            cur_masses = self.model.body_mass[idx, bid]  # shape (B,)

            # small perturbation
            rng, subrng = jax.random.split(rng)
            noise = 0.05 * jax.random.normal(subrng, (B,))
            new_masses = jnp.clip(cur_masses + noise, 0.1, 0.75)
            new_weights = jnp.ones((B,), dtype=jnp.float32) / B  # reset to uniform
        else:
            new_masses = self.model.body_mass[:, bid]
            new_weights = w  # keep current weights
        # update domain weights
        self.domain_weights = new_weights

        # write back into the body_mass array
        new_body_mass = self.model.body_mass.at[:, bid].set(new_masses)

        # IMPORTANT: replace the field on the model, don't assign the array to the model
        self.model = self.model.tree_replace({"body_mass": new_body_mass})
        
        self.risk_strategy.set_weights(self.domain_weights)

        return ess < 0.9 * B, new_masses  # whether resampling was done