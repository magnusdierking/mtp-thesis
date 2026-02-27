import argparse

from hydrax.algs import MPPI, MTP, CEM
from hydrax.algs.mtp.an_mtp_opt import AnMTP
# from hydrax.algs.mtp.an_mtp_dr import AnMTP

from hydrax.utils.files import get_data_path
from hydrax.simulation.deterministic import run_interactive
# from hydrax.simulation.deterministic_dr import run_interactive
from hydrax.simulation.deterministic_headless import run_headless_simulation

from hydrax.tasks.pusht_franka import PushTFranka
import jax
import jax.numpy as jnp
import numpy as np



# --------------------------------------------------- #
UPDATE_COV = False
NUM_SAMPLES = 512
NUM_RANDOMIZATIONS = 1
PLANNING_FREQUENCY = 20
PLANNING_HORIZON = 10
SIM_STEPS_PER_CONTROL_STEP = 2
MAX_SPEED = 0.35  # m/s

SIGMA = 0.2
ALPHA = 0.1
TEMPERATURE = 0.1
NUM_ELITES = 36

SEED = 42
# ------------------------------------------------- #



# seed = 100
# update_cov = True
# sigma_max = 0.75
# sigma_min = 0.05
# sigma_start = 0.2
# det_init = {
#     "block_pos_x": -0.1,
#     "block_pos_y": 0.15,
#     "block_angle": np.pi/4,
#     "ee_goal_pos": [0.35, 0.0, 0.035]
# }

# seed = 200
# update_cov = False
# sigma_max = 0.75
# sigma_min = 0.05
# sigma_start = 0.2
# det_init = {
#     "block_pos_x": 0.05,
#     "block_pos_y": 0.15,
#     "block_angle": 3*np.pi/4,
#     "ee_goal_pos": [0.45, 0.1, 0.035]
# }


# det_init = {
#     "block_pos_x": 0.6,
#     "block_pos_y": 0.05,
#     "block_angle": np.pi/8,
#     "ee_goal_pos": [0.4, 0.0, 0.045]
# }


det_init = {
    "block_pos_x": 0.6,
    "block_pos_y": -0.1,
    "block_angle": np.pi/4,
    "ee_goal_pos": [0.45, 0.1, 0.035]
}


# det_init = {
#     "block_pos_x": 0.6,
#     "block_pos_y": -0.2,
#     "block_angle": 0,
#     "ee_goal_pos": [0.4, 0.0, 0.045]
# }


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
subparsers.add_parser("anmtp", help="Annealed MTP")
args = parser.parse_args()


path = get_data_path() / "pushT_sim_sweep" / "free"
if not path.exists():       
    path.mkdir(parents=True, exist_ok=True)
path = path / f"seed_{SEED}_update_cov_{UPDATE_COV}.csv"


# Set the controller based on command-line arguments
if args.algorithm is None: 
    args.algorithm = "mtp"  # Default to MTP

elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        temperature=0.1,
        num_samples=NUM_SAMPLES,
        M=4, # horizon via control points
        N=32, # samples
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_start=sigma_start,
        num_elites=36,
        keep_elites=1,   # !experimental
        beta=0.3,
        alpha=0.1,
        interpolation='bspline',
        num_randomizations=NUM_RANDOMIZATIONS,
        seed=seed,
        update_cov=update_cov,
        planning_freq=20,
        savgol_filter=True,  # !experimental
        default_zero_controls=False,
    )
    error_log = "./../data/error_log_pushT/mtp_{seed}.npy".format(seed=seed)
    
elif args.algorithm == "anmtp":
    print("Running AnMTP")
    ctrl = AnMTP(
            task,
            num_samples=NUM_SAMPLES,
            M=3, # horizon via control points
            N=64, # samples
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_start=sigma_start,
            num_elites=24,
            keep_elites=3,   # !experimental
            beta = 0.25,
            beta_lr = 0.1,        # adaptation step size
            beta_min = 0.0,
            beta_max = 0.35,
            alpha=0.0,
            interpolation='bspline',
            shift = True,
            num_randomizations=NUM_RANDOMIZATIONS,
            seed=seed,
            update_cov=update_cov,
        )
    error_log = "./../data/error_log_pushT/anmtp_{seed}.npy".format(seed=seed)
    







mj_model, mj_data = task.reset(seed=seed)

# Run the interactive simulation
max_traces = 20
trace_idxs = [i * max_traces for i in range(NUM_SAMPLES // max_traces)]
print("Tracing indices:", trace_idxs)
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=20,
    show_traces=True,
    trace_width=0.25,
    max_traces=max_traces,
    fixed_camera_id=0,
    show_ui=True,
    record_video=False,
    max_step=300,
    seed=seed,
    trace_idxs=trace_idxs,
    # log_file=path.as_posix(),
    )


# run_headless_simulation(
#     task,
#     ctrl,
#     frequency=50,
#     seeds=[seed],
#     max_step=500,
#     log_file_prefix="pusht_franka_" + args.algorithm,
#     save_path="./results"
#     )