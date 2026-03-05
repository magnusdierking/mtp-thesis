import argparse
from random import seed

import jax.numpy as jnp
import numpy as np

from hydrax.algs import CEM, MPPI, MTP, PredictiveSampling
from hydrax.simulation.deterministic_beta import run_interactive
from hydrax.tasks.pusht_franka import PushTFranka
from hydrax.utils.files import get_data_path

# --------------------------------------------------- #
UPDATE_COV = False
SHIFT = True
SAVGOL_FILTER = False,
NUM_SAMPLES = 256
NUM_RANDOMIZATIONS = 1
PLANNING_FREQUENCY = 20

# short horizon 15 x 1, long horizon 25 x 1
PLANNING_HORIZON = 20
SIM_STEPS_PER_CONTROL_STEP = 1
MAX_SPEED = 0.35  # m/s

SIGMA = 0.2
ALPHA = 0.1
TEMPERATURE = 0.1
NUM_ELITES = 48
KEEP_ELITES = 1

# AnMTP
BETA = 0.4

SEED = 40
# ------------------------------------------------- #


if SEED == 40:
    det_init = {
        "block_pos_x": 0.6,
        "block_pos_y": -0.1,
        "block_angle": np.pi/2,
        "ee_goal_pos": [0.45, 0.1, 0.035]
    }
elif SEED == 41:
    det_init = {
        "block_pos_x": 0.65,
        "block_pos_y": 0.0,
        "block_angle": np.pi,
        "ee_goal_pos": [0.45, 0.1, 0.035]
    }
elif SEED == 42:
    det_init = {
        "block_pos_x": 0.6,
        "block_pos_y": 0.05,
        "block_angle": 5*np.pi/4,
        "ee_goal_pos": [0.45, 0.1, 0.035]
    }
elif SEED == 43:
    det_init = {
        "block_pos_x": 0.5,
        "block_pos_y": 0.2,
        "block_angle": -np.pi/2,
        "ee_goal_pos": [0.45, 0.1, 0.035]
    }
elif SEED == 44:
    det_init = {
        "block_pos_x": 0.4,
        "block_pos_y": -0.05,
        "block_angle": -np.pi/4,
        "ee_goal_pos": [0.45, 0.1, 0.035]
    }
elif SEED == 45:
    det_init = {
        "block_pos_x": 0.45,
        "block_pos_y": -0.15,
        "block_angle": 0*np.pi,
        "ee_goal_pos": [0.45, 0.1, 0.035]
    }



task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=PLANNING_HORIZON,
                    sim_steps_per_control_step=SIM_STEPS_PER_CONTROL_STEP,
                    ctrl_limits={"u_min": jnp.array([-MAX_SPEED, -MAX_SPEED]), 
                                 "u_max": jnp.array([MAX_SPEED, MAX_SPEED])},
                    trace_sites=["ee_site"],
                    actuation_type='velocity',
                    sampling_space="velocity",
                    det_init=det_init,
                    block_type = 'free',
                )

parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the walker task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross Entropy Method")
subparsers.add_parser("mtp", help="MTP")
subparsers.add_parser("ps", help="Predictive Sampling")
args = parser.parse_args()



if args.algorithm is None: 
    args.algorithm = "mtp"  # Default to MTP
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=SIGMA,
        temperature=TEMPERATURE,
        num_randomizations=NUM_RANDOMIZATIONS,
        alpha=ALPHA,
        seed=SEED,
        update_cov=UPDATE_COV,
        shift=SHIFT,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=NUM_SAMPLES,
        num_elites=NUM_ELITES,
        sigma_start=SIGMA,
        sigma_min=SIGMA,
        sigma_max=SIGMA,
        alpha=ALPHA,
        num_randomizations=NUM_RANDOMIZATIONS,
        seed=SEED,
        shift=SHIFT,
        planning_freq=PLANNING_FREQUENCY,
        update_cov=UPDATE_COV,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        temperature=TEMPERATURE,
        num_samples=NUM_SAMPLES,
        M=3,
        N=64,
        sigma_min=SIGMA,
        sigma_max=SIGMA,
        sigma_start=SIGMA,
        num_elites=NUM_ELITES,
        keep_elites=KEEP_ELITES,   
        beta=BETA,
        alpha=ALPHA,
        interpolation='akima',
        num_randomizations=NUM_RANDOMIZATIONS,
        seed=SEED,
        update_cov=UPDATE_COV,
        shift=SHIFT,
        savgol_filter=SAVGOL_FILTER,
        planning_freq=PLANNING_FREQUENCY,
    )
elif args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=NUM_SAMPLES,
        num_randomizations=NUM_RANDOMIZATIONS,
        seed=SEED,
        noise_level=SIGMA,
        shift=SHIFT,
        alpha=ALPHA,
        savgol_filter=SAVGOL_FILTER,
        planning_freq=PLANNING_FREQUENCY,
    )



mj_model, mj_data = task.reset(seed=SEED)

path = get_data_path() / "pushT_sim_sweep" / "free_long"
if not path.exists():       
    path.mkdir(parents=True, exist_ok=True)

path = path / f"seed_{SEED}_{args.algorithm}.pkl"

# Run the interactive simulation
max_traces = 30
trace_idxs = [i * max_traces for i in range(NUM_SAMPLES // max_traces)]
print("Tracing indices:", trace_idxs)
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=PLANNING_FREQUENCY,
    show_traces=True,
    trace_width=0.25,
    max_traces=max_traces,
    fixed_camera_id=0,
    show_ui=True,
    record_video=False,
    max_step=100,
    seed=SEED,
    trace_idxs=trace_idxs,
    log_file=path.as_posix(),
    )