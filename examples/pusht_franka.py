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

"""
Run an interactive simulation of the push-T task with predictive sampling.
"""

# Define the task (cost and dynamics)
#velocity control
task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=10,
                    sim_steps_per_control_step=5,
                    ctrl_limits={"u_min": jnp.array([-0.4, -0.4]), 
                                "u_max": jnp.array([0.4, 0.4])},
                    trace_sites=["ee_site"],
                    actuation_type='velocity',
                    sampling_space="velocity",
                )

# position control
# task = PushTFranka(ik_type = 'pinv',
#                 planning_horizon=15,
#                 sim_steps_per_control_step=5,
#                 ctrl_limits={"u_min": jnp.array([0.35, 0.8]), 
#                             "u_max": jnp.array([-0.35, 0.35])},
#                 trace_sites=["ee_site"],
#                 actuation_type='position',
#                 sampling_space="position",
#                 )

# Parse command-line arguments
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



seed = 845545 # 36, ... 
num_samples = 512
num_randomizations = 1

data = {}
path = get_data_path() / "pushT_sim" / args.algorithm 
if not path.exists():       
    path.mkdir(parents=True, exist_ok=True)
path = path / f"seed_{seed}.csv"


# Set the controller based on command-line arguments
if args.algorithm is None: 
    args.algorithm = "mtp"  # Default to MTP
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=num_samples,
        noise_level=0.25,
        temperature=0.1,
        num_randomizations=num_randomizations,
        colorize_noise=False,   # !experimental
        alpha=0.1,
        seed=seed
    )
    error_log = "./../data/error_log_pushT/mppi_{seed}.npy".format(seed=seed)
    
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=num_samples,
        num_elites=12,
        sigma_start=0.2,
        sigma_min=0.05,
        alpha=0.1,
        num_randomizations=num_randomizations,
    )
    error_log = "./../data/error_log_pushT/cem_{seed}.npy".format(seed=seed)
    
elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=num_samples,
        M=3, # horizon via control points
        N=32, # samples 
        sigma_min=0.1,
        sigma_start=0.2,
        num_elites=12,
        beta=0.35,
        alpha=0.1,
        interpolation='bspline',
        num_randomizations=num_randomizations,
        seed=seed,
    )
    error_log = "./../data/error_log_pushT/mtp_{seed}.npy".format(seed=seed)
    
elif args.algorithm == "anmtp":
    print("Running AnMTP")
    ctrl = AnMTP(
            task,
            num_samples=num_samples,
            M=3, # horizon via control points
            N=32, # samples 
            sigma_min=0.05,
            sigma_start=0.2,
            num_elites=24,
            keep_elites=4,   # !experimental
            beta = 0.25,
            beta_lr = 0.1,        # adaptation step size
            beta_min = 0.05,
            beta_max = 0.35,
            alpha=0.2,
            interpolation='akima',
            shift = True,
            num_randomizations=num_randomizations,
            seed=seed,
        )
    error_log = "./../data/error_log_pushT/anmtp_{seed}.npy".format(seed=seed)
    
# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)

# Run the interactive simulation

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=10,
    show_traces=True,
    trace_width=0.55,
    max_traces=32,
    fixed_camera_id=0,
    show_ui=True,
    record_video=False,
    max_step=500,
    seed=seed,
    log_file=path.as_posix(),
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