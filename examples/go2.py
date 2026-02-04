import argparse

from hydrax.algs import MPPI, MTP
from hydrax.algs.mtp.an_mtp_opt import AnMTP
from hydrax.files import get_root_path
from hydrax.simulation.deterministic import run_interactive
from hydrax.simulation.deterministic_headless import run_headless_simulation
from hydrax.tasks.go2 import Go2VelocityTask


# Define the task (cost and dynamics)
task = Go2VelocityTask(xml_path=(get_root_path() / "models" / "unitree_go2" / "scene_mjx.xml").as_posix(),
                       actuation_type='velocity',
                       sim_steps_per_control_step=2,
                       planning_horizon=10,  
                       target_vx = 0.2, # m/s
                       target_vy = 0.0, # m/s
                       target_yaw_rate = 0.0, # rad/s
                       target_height = 0.445,)

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

seed = 25

# Set the controller based on command-line arguments
if args.algorithm is None: 
    args.algorithm = "mtp"  # Default to MTP
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.1,
        temperature=0.1,
        shift=True,
        planning_freq=10,
        num_randomizations=1,
        seed=seed,
    )
elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
            task,
            num_samples=128,
            M=3, # horizon via control points
            N=16, # samples 
            sigma_min=0.1,
            sigma_start=0.2,
            num_elites=10,
            beta=0.25,
            alpha=0.1,
            interpolation='bspline',
            num_randomizations=2,
            seed=seed,
        )
elif args.algorithm == "anmtp":
    print("Running AnMTP")
    ctrl = AnMTP(
            task,
            num_samples=128,
            M=3, # horizon via control points
            N=16, # samples 
            sigma_min=0.05,
            sigma_start=0.2,
            num_elites=10,
            beta = 0.15,
            beta_lr = 0.05,        # adaptation step size
            beta_min = 0.1,
            beta_max = 0.3,
            alpha=0.1,
            interpolation='bspline',
            num_randomizations=2,
            seed=seed,
        )
# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)

# Run the interactive simulation

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=10,
    show_traces=True,
    trace_width=0.75,
    max_traces=10,
    fixed_camera_id=0,
    show_ui=True,
    record_video=False,
    seed=seed,
    )
