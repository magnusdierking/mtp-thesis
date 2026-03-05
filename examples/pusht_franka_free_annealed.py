import argparse
from random import seed

from hydrax.algs import MPPI, MTP, CEM
from hydrax.algs.mtp.an_mtp_opt import AnMTP
# from hydrax.algs.mtp.an_mtp_dr import AnMTP

from hydrax.utils.files import get_data_path
from hydrax.simulation.deterministic_beta import run_interactive
from hydrax.simulation.deterministic_headless import run_headless_simulation

from hydrax.tasks.pusht_franka import PushTFranka
import jax
import jax.numpy as jnp
import numpy as np



# --------------------------------------------------- #
UPDATE_COV = False
SHIFT = True
SAVGOL_FILTER = False,
NUM_SAMPLES = 256
NUM_RANDOMIZATIONS = 1
PLANNING_FREQUENCY = 20
PLANNING_HORIZON = 15
SIM_STEPS_PER_CONTROL_STEP = 1
MAX_SPEED = 0.35  # m/s

SIGMA = 0.2
ALPHA = 0.1
TEMPERATURE = 0.1
NUM_ELITES = 48
KEEP_ELITES = 1

# AnMTP
BETA = 0.4
BETA_MIN = 0.2
BETA_MAX = 0.6

SEED = 44
TYPE = 'free' # '3dof' or 'free'
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

elif SEED == 48:
    det_init = {
        "block_pos_x": 0.6,
        "block_pos_y": -0.1,
        "block_angle": np.pi/8,
        "ee_goal_pos": [0.45, 0.1, 0.035]
    }

if TYPE == '3dof':
    det_init["block_pos_x"] = det_init["block_pos_x"] - 0.5 - 0.09

task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=PLANNING_HORIZON,
                    sim_steps_per_control_step=SIM_STEPS_PER_CONTROL_STEP,
                    ctrl_limits={"u_min": jnp.array([-MAX_SPEED, -MAX_SPEED]), 
                                 "u_max": jnp.array([MAX_SPEED, MAX_SPEED])},
                    trace_sites=["ee_site"],
                    actuation_type='velocity',
                    sampling_space="velocity",
                    det_init=det_init,
                    block_type = TYPE,
                )

parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the walker task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("mtp", help="MTP")
subparsers.add_parser("anmtp", help="Annealed MTP")
args = parser.parse_args()



if args.algorithm is None: 
    args.algorithm = "mtp"  # Default to MTP
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
elif args.algorithm == "anmtp":
    print("Running AnMTP")
    ctrl = AnMTP(
        task,
        num_samples=NUM_SAMPLES,
        M=3, # horizon via control points
        N=64, # samples
        sigma_min=SIGMA,
        sigma_max=SIGMA,
        sigma_start=SIGMA,
        num_elites=NUM_ELITES,
        keep_elites=KEEP_ELITES,   # !experimental
        beta = BETA,
        beta_min = BETA_MIN,
        beta_max = BETA_MAX,
        alpha=ALPHA,
        interpolation='bspline',
        savgol_filter=SAVGOL_FILTER,
        shift = SHIFT,
        num_randomizations=NUM_RANDOMIZATIONS,
        seed=SEED,
        update_cov=UPDATE_COV ,
    )


mj_model, mj_data = task.reset(seed=SEED)

path = get_data_path() / "pushT_sim_extensions" / "annealing"
if not path.exists():       
    path.mkdir(parents=True, exist_ok=True)
if args.algorithm in ["mtp"]:
    path = path / f"seed_{SEED}_{args.algorithm}_{BETA}.pkl"
else:    
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