import argparse

from hydrax.algs import MPPI, MTP
# from hydrax.algs.mtp.an_mtp_opt import AnMTP
from hydrax.algs.mtp.an_mtp_dr import AnMTP
from hydrax.simulation.deterministic_dr import run_interactive
from hydrax.simulation.deterministic_headless import run_headless_simulation
from hydrax.tasks.pusht_franka import PushTFranka

"""
Run an interactive simulation of the push-T task with predictive sampling.
"""

# Define the task (cost and dynamics)
task = PushTFranka(ik_type = 'pinv')

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the walker task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("mtp", help="MTP")
subparsers.add_parser("anmtp", help="Annealed MTP")
args = parser.parse_args()

seed = 34

# Set the controller based on command-line arguments
if args.algorithm is None: 
    args.algorithm = "mtp"  # Default to MTP
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.3,
        temperature=0.1,
        num_randomizations=4,
        seed=seed,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
            task,
            num_samples=128,
            M=2, # horizon via control points
            N=16, # samples 
            sigma_min=0.1,
            sigma_start=0.2,
            num_elites=10,
            beta=0.25,
            alpha=0.1,
            interpolation='bspline',
            num_randomizations=5,
            seed=seed,
        )
elif args.algorithm == "anmtp":
    print("Running AnMTP")
    ctrl = AnMTP(
            task,
            num_samples=64,
            M=2, # horizon via control points
            N=16, # samples 
            sigma_min=0.05,
            sigma_start=0.1,
            num_elites=12,
            keep_elites=2,   # !experimental
            beta = 0.3,
            beta_lr = 0.1,        # adaptation step size
            beta_min = 0.05,
            beta_max = 0.35,
            alpha=0.05,
            interpolation='bspline',
            num_randomizations=8,
            seed=seed,
        )
# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)

# Run the interactive simulation

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=25,
    show_traces=True,
    trace_width=0.55,
    max_traces=16,
    fixed_camera_id=0,
    show_ui=True,
    record_video=False,
    seed=seed,
    )

# run_headless_simulation(
#     task,
#     ctrl,
#     frequency=50,
#     seeds=[seed],
#     max_step=1000,
#     log_file_prefix="pusht_franka_" + args.algorithm,
#     save_path="./results"
#     )