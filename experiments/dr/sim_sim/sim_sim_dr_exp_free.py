import argparse
import sys
from pathlib import Path

import mujoco

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from an_mtp_dr import AnMTP
from deterministic_dr_ghost import run_interactive
from domain_adaptation import UniformDomainRandomization
from domain_randomization_utils import (
    compute_randomizations,
    compute_randomizations_with_derived_quantities,
)
from mujoco import mjx

from hydrax.algs import CEM, MPPI, MTP, PredictiveSampling
from hydrax.risk import (
    AverageCost,
    ConditionalValueAtRisk,
    ExpectedCost,
    InverseConditionalValueAtRisk,
    InverseValueAtRisk,
    RiskStrategy,
    ValueAtRisk,
)
from hydrax.tasks.pusht_franka import PushTFranka
from hydrax.utils.files import get_data_path, get_root_path

NUM_SAMPLES = 32
NUM_RANDOMIZATIONS = 24
MAX_SPEED = 0.35  # m/s

seed = 42
update_cov = False
sigma_max = 0.35
sigma_min = 0.15
sigma_start = 0.2


det_init = {
    "block_pos_x": 0.6,
    "block_pos_y": 0.1,
    "block_angle": 6 * np.pi / 5,
    "ee_goal_pos": [0.5, -0.1, 0.035],
}


task = PushTFranka(
    ik_type="pinv",
    planning_horizon=10,
    sim_steps_per_control_step=2,
    ctrl_limits={
        "u_min": jnp.array([-MAX_SPEED, -MAX_SPEED]),
        "u_max": jnp.array([MAX_SPEED, MAX_SPEED]),
    },
    trace_sites=["T_1", "T_2", "ee_site", "T_3", "block_site"],
    actuation_type="velocity",
    sampling_space="velocity",
    det_init=det_init,
    block_type="dr-free",
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
    help="Domain randomization strategy",
)
parser.add_argument(
    "--risk",
    choices=["average", "expectation", "var", "cvar", "ivar", "icvar"],
    help="risk aggregation method for domain randomization",
)

args = parser.parse_args()
print(args)

risk_alpha = 0.5  # for (C)VaR
uniform_weights = (
    jnp.ones((NUM_RANDOMIZATIONS,), dtype=jnp.float32) / NUM_RANDOMIZATIONS
)

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
    aggregation = InverseConditionalValueAtRisk(
        alpha=risk_alpha, weights=uniform_weights
    )


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


path = get_data_path() / "dr_sim"
if not path.exists():
    path.mkdir(parents=True, exist_ok=True)
path = path / f"seed_{seed}_{args.algorithm}_{args.dr}_{args.risk}"


randomization_dict = {
    "geoms": {
        "ground": {
            "geom_friction": (0, jnp.linspace(0.1, 5.0, NUM_RANDOMIZATIONS)),
        },
        # 'bottom': {
        #     'geom_mass': (0, jnp.linspace(0.01, 1.5, NUM_RANDOMIZATIONS)),
        # },
        # 'top': {
        #     'geom_mass': (0, jnp.linspace(0.05, 1.0, NUM_RANDOMIZATIONS)),
        # },
    },
    # 'bodies': {
    #     'block': {
    #         'body_mass': (None, jnp.linspace(0.1, 1.5, NUM_RANDOMIZATIONS)),
    #         # 'body_ipos': (0, jnp.linspace(0.0, 0.08, NUM_RANDOMIZATIONS)),
    #     },
    # }
}
# print(dir(ctrl.model))
# sys.exit(0)
# Define mass values for each geom
top_masses = jnp.linspace(0.1, 0.11, NUM_RANDOMIZATIONS)
bottom_masses = jnp.linspace(0.01, 0.5, NUM_RANDOMIZATIONS)

body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "block")

randomized_axes = [
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_invweight0",
    "dof_invweight0",
    "dof_M0",
    "dof_armature",
    "body_subtreemass",
]

derived_values = {field: [] for field in randomized_axes}

for top_mass, bottom_mass in zip(top_masses, bottom_masses):
    spec = mujoco.MjSpec.from_file(
        (
            get_root_path()
            / "hydrax"
            / "models"
            / "fr3_pushT_vel"
            / "scene_mjx_free_dr.xml"
        ).as_posix()
    )
    body = spec.body("block")

    # Set mass for each geom by name
    for geom in body.geoms:
        if geom.name == "top":
            geom.mass = float(top_mass)
            print(f"Set 'top' geom mass: {top_mass}")
        elif geom.name == "bottom":
            geom.mass = float(bottom_mass)
            print(f"Set 'bottom' geom mass: {bottom_mass}")

    compiled_model = spec.compile()
    total_mass = compiled_model.body_mass[body_id]
    print(
        f"Compiled body mass (top={top_mass:.3f} + bottom={bottom_mass:.3f}): {total_mass:.3f}"
    )
    print(f"Compiled inertia: {compiled_model.body_inertia[body_id]}")
    print(f"Compiled ipos: {compiled_model.body_ipos[body_id]}\n")

    # Append derived quantities to lists
    derived_values["body_mass"].append(compiled_model.body_mass)
    derived_values["body_inertia"].append(compiled_model.body_inertia)
    derived_values["body_ipos"].append(compiled_model.body_ipos)
    derived_values["body_invweight0"].append(compiled_model.body_invweight0)
    derived_values["dof_invweight0"].append(compiled_model.dof_invweight0)
    derived_values["dof_M0"].append(compiled_model.dof_M0)
    derived_values["dof_armature"].append(compiled_model.dof_armature)
    derived_values["body_subtreemass"].append(compiled_model.body_subtreemass)

# Turn all lists in derived_values into arrays
for field in derived_values:
    derived_values[field] = jnp.array(derived_values[field])

print("\n=== Final Randomized Values ===")
for field, values in derived_values.items():
    print(f"{field} (block body): {values[:, body_id]}")


ctrl.update_randomized_axes(randomized_axes)

ctrl.model = ctrl.model.replace(**derived_values)

# ctrl.model, randomized_axes = compute_randomizations(
#     ctrl.model,
#     mj_model,
#     randomization_dict,
# )

# sys.exit(0)

# ctrl.model, randomized_axes = compute_randomizations_with_derived_quantities(
#     ctrl.model,
#     mj_model,
#     (get_root_path() / "hydrax" / "models" / "fr3_pushT_vel" / "scene_mjx_free_dr.xml").as_posix(),
#     randomization_dict,
# )


max_traces = 128
trace_idxs = (
    [i * max_traces for i in range(NUM_SAMPLES // max_traces)] if max_traces > 0 else []
)
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
    # dr_strategy = dr_strategy,
    trace_idxs=trace_idxs,
)
