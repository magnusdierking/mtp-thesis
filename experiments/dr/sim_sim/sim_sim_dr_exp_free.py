import argparse, sys
from pathlib import Path

import mujoco

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from hydrax.algs import MPPI, MTP, CEM, PredictiveSampling
# from hydrax.algs.mtp.an_mtp_opt import AnMTP
from an_mtp_dr import AnMTP
from mujoco import mjx
from hydrax.utils.files import get_data_path
from deterministic_dr_ghost import run_interactive

from hydrax.tasks.pusht_franka import PushTFranka
import jax.numpy as jnp
import numpy as np

from hydrax.risk import RiskStrategy, ExpectedCost, AverageCost, ValueAtRisk, ConditionalValueAtRisk,InverseValueAtRisk, InverseConditionalValueAtRisk
from domain_adaptation import UniformDomainRandomization


import jax
import jax.numpy as jnp
from typing import Dict, Optional
from domain_randomization_utils import compute_randomizations


def apply_domain_randomization(
    model: mjx.Model,
    field: str,
    values: jnp.ndarray,
    *,
    component_idx: Optional[int] = None,
) -> mjx.Model:
    """
    Args:
      model: mjx.Model or a batched mjx.Model.
      field: Name of the mjx.Model attribute to change, e.g. "geom_friction" or "geom_solref".
      values: Array of new values, one per domain.
              Shape:
                - if you are fully replacing the field: (batch, *field.shape[1:])
                - if you are replacing just one component: (batch, *field.shape[2:])
      component_idx: If not None, replace only this component of the last axis
                     (e.g. 0 or 1 for solref, 0/1/2 for geom_friction).
                     If None, the entire field is replaced.

    Returns:
      A new mjx.Model with the updated field.
    """
    # Get the original field
    orig = getattr(model, field)

    # If the model is unbatched and you want to batch only this field, add batch dim
    # (commonly you instead create a batched model externally via vmap).
    if values.ndim == orig.ndim + 1:
        # values has batch dim, orig does not
        # Example: orig: (ngeom, 3), values: (batch, ngeom, 3)
        new_field = values
    else:
        # Assume orig already has a batch dimension: (batch, ...)
        if component_idx is None:
            # Replace entire field: shapes must match
            new_field = values
        else:
            # Replace a single component along the last axis
            # orig: (batch, ..., C), values: (batch, ..., )
            # or values: (batch, ..., 1) which we squeeze
            v = values
            if v.shape[-1] == 1:
                v = jnp.squeeze(v, axis=-1)
            # Broadcast v to match orig except last dimension
            # We rely on JAX broadcasting for any trailing dimensions.
            new_field = orig.at[..., component_idx].set(v)

    # Use tree_replace/replace to build a new model
    # For current mjx versions, Model is a dataclass-like PyTree, so replace works:
    return model.replace(**{field: new_field})


NUM_SAMPLES = 64
NUM_RANDOMIZATIONS = 6  
MAX_SPEED = 0.35  # m/s

seed = 42
update_cov = False
sigma_max = 0.35
sigma_min = 0.15
sigma_start = 0.2
# joint
# det_init = {
#     "block_pos_x": 0.1,
#     "block_pos_y": 0.15,
#     "block_angle": np.pi/3,
#     "ee_goal_pos": [0.5, -0.0, 0.035]
# }
#free hard
# det_init = {
#     "block_pos_x": 0.4,
#     "block_pos_y": 0.15,
#     "block_angle": np.pi/3,
#     "ee_goal_pos": [0.4, 0.0, 0.035]
# }
# free easy
det_init = {
    "block_pos_x": 0.45,
    "block_pos_y": 0.15,
    "block_angle": np.pi/4,
    "ee_goal_pos": [0.35, 0.25, 0.035]
}


task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=10,
                    sim_steps_per_control_step=2,
                    ctrl_limits={"u_min": jnp.array([-MAX_SPEED, -MAX_SPEED]), 
                                 "u_max": jnp.array([MAX_SPEED, MAX_SPEED])},
                    trace_sites=["T_1", "T_2","ee_site", "T_3", "block_site"],
                    actuation_type='velocity',
                    sampling_space="velocity",
                    det_init=det_init,
                    block_type = 'dr-free',
                )


parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the push t task."
)

# Algorithm subparser group
algorithm_subparsers = parser.add_subparsers(
    dest="algorithm", required=True, help="Sampling algorithm (choose one)"
)
algorithm_subparsers.add_parser("mppi")
algorithm_subparsers.add_parser("cem")
algorithm_subparsers.add_parser("mtp")
algorithm_subparsers.add_parser("ps")

# Domain randomization argument (normal argument, not subparser)
parser.add_argument(
    "--dr",
    choices=["uniform", "evolutionary", "bayesian"],
    help="Domain randomization strategy"
)
parser.add_argument(
    "--risk",
    choices=["average", "expectation", "var", "cvar", "ivar", "icvar"],
    help="risk aggregation method for domain randomization"
)

args = parser.parse_args()
print(args)

risk_alpha = 0.5  # for (C)VaR
uniform_weights = jnp.ones((NUM_RANDOMIZATIONS,), dtype=jnp.float32) / NUM_RANDOMIZATIONS

if args.risk is None or args.risk == "average": 
    args.risk = "average"  
    aggregation = AverageCost()
elif args.risk == "expectation":
    # initialize with uniform weights
    aggregation = ExpectedCost(weights=uniform_weights)
elif args.risk == "var":
    aggregation = ValueAtRisk(alpha=risk_alpha)
elif args.risk == "cvar":
    aggregation = ConditionalValueAtRisk(alpha=risk_alpha, weights=uniform_weights)
elif args.risk == "ivar":
    aggregation = InverseValueAtRisk(alpha=risk_alpha)
elif args.risk == "icvar":
    aggregation = InverseConditionalValueAtRisk(alpha=risk_alpha, weights=uniform_weights)


if args.algorithm is None:
    args.algorithm = "mtp"  # Default to MTP
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=0.2,
        temperature=0.1,
        num_randomizations=NUM_RANDOMIZATIONS,
        colorize_noise=False,  # !experimental
        alpha=0.1,
        seed=seed,
        update_cov=update_cov,
    )
    error_log = f"./../data/error_log_pushT/mppi_{seed}.npy"

elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=NUM_SAMPLES,
        num_elites=12,
        sigma_start=sigma_start,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        alpha=0.1,
        shift=True,
        planning_freq=5,
        num_randomizations=NUM_RANDOMIZATIONS,
        seed=seed,
        update_cov=update_cov,
    )
    error_log = "./../data/error_log_pushT/cem_{seed}.npy".format(seed=seed)

elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=NUM_SAMPLES,
        M=3,  # horizon via control points
        N=64,  # samples
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_start=sigma_start,
        num_elites=12,
        beta=0.25,
        alpha=0.1,
        interpolation="bspline",
        num_randomizations=NUM_RANDOMIZATIONS,
        seed=seed,
        update_cov=update_cov,
        planning_freq=5,
    )
    error_log = f"./../data/error_log_pushT/mtp_{seed}.npy"

elif args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=0.2,
        num_randomizations=NUM_RANDOMIZATIONS,
        shift=True,
        planning_freq=5,
        risk_strategy=aggregation,
        savgol_filter=True,
        seed=seed,
        alpha=0.1,
    )
    error_log = f"./../data/error_log_pushT/ps_{seed}.npy"
    

    
# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)

print("Using Uniform Domain Randomization.")
dr_strategy = UniformDomainRandomization(
    seed=seed,
    task=task,
    controller=ctrl,
    randomized_bodies={#"ground": {"field": "geom_friction", "min": [0.1], "max": [5.0], "internal_idx": [0]},
                    #    "bottom": {"field": "body_mass", "min": [0.1], "max": [5.0], "internal_idx": [0]},
                    #    "top": {"field": "body_mass", "min": [0.1], "max": [5.0], "internal_idx": [0]},
                    #    "bottom": {"field": "geom_mass", "min": [0.1], "max": [5.0], "internal_idx": [0]},
                    #    "ee": {"field": "geom_friction", "min": [0.1], "max": [5.0], "internal_idx": [0]},
                    #    "ground": {"field": "geom_friction", "min": [0.1], "max": [5.0], "internal_idx": [0]},
                    #  "ground": {"field": "geom_solimp", "min": [0.002], "max": [0.3], "internal_idx": [2]},
                        # "top": {"field": "geom_solref", "min": [0.1], "max": [3], "internal_idx": [1]},                        
    },
    randomized_joints = {},
    num_randomizations=NUM_RANDOMIZATIONS,
)

# "bottom": {"field": "geom_friction", "min": [0.0001], "max": [10], "internal_idx": [0]},
# "top": {"field": "geom_friction", "min": [0.0001], "max": [10], "internal_idx": [0]},
# "ground": {"field": "geom_friction", "min": [0.0001], "max": [10], "internal_idx": [0]},

path = get_data_path() / "dr_sim"  
if not path.exists():       
    path.mkdir(parents=True, exist_ok=True)
path = path / f"seed_{seed}_{args.algorithm}_{args.dr}_{args.risk}"



for i in range(mj_model.nbody):
    name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"Body {i}: {name}")
    
for i in range(mj_model.ngeom):
    name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
    print(f"Geom {i}: {name}")

for i in range(mj_model.njnt):
    name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"Joint {i}: {name}")




randomization_dict = {
    'geoms': {
        'ground': {
            'geom_solimp': (2, jnp.linspace(0.002, 0.3, NUM_RANDOMIZATIONS)),  
            'geom_friction': (None, jnp.linspace(0.1, 5.0, NUM_RANDOMIZATIONS)), 
        },
    },
    # 'bodies': {
    #     'block': {
    #         'body_mass': (None, jnp.linspace(0.1, 0.5, NUM_RANDOMIZATIONS)),
    #     },
    # }
}



ctrl.model, randomized_axes = compute_randomizations(
    ctrl.model,
    mj_model,
    randomization_dict,
)
ctrl.update_randomized_axes(randomized_axes)





max_traces = 128
trace_idxs = [i * max_traces for i in range(NUM_SAMPLES // max_traces)] if max_traces > 0 else []
print("Tracing indices:", trace_idxs)

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=5,
    show_traces=True,
    trace_width=0.55,
    max_traces=max_traces,
    fixed_camera_id=0,
    show_ui=True,
    record_video=False,
    max_step=500,
    seed=seed,
    # log_file=path.as_posix(),
    dr_strategy = dr_strategy,
    trace_idxs=trace_idxs,
    )
