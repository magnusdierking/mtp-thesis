import argparse

from hydrax.algs import MTP
from hydrax.algs import AnMTP
from hydrax.algs import MPPI, Evosax, PredictiveSampling
from hydrax.algs import CEM
from hydrax.risk import WorstCase
from hydrax.simulation.deterministic import run_interactive

from hydrax.tasks.bugtrap import BugTrap


"""
Run an interactive simulation of the particle tracking task.

Double click on the green target, then drag it around with [ctrl + right-click].
"""

# Define the task (cost and dynamics)
task = BugTrap()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the bug trap task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="CEM")
subparsers.add_parser("mtp", help="MTP")
subparsers.add_parser("anmtp", help="Annealed MTP")
args = parser.parse_args()

# Set the controller based on command-line arguments
if args.algorithm is None: 
    args.algorithm = "mtp"  # Default to MTP

elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(task, 
                num_samples=128, 
                noise_level=0.35, 
                temperature=0.01)
    # save_path = "./../data/headless_bugtrap_mppi"
    save_path = "./../data/test"

elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=8,
        sigma_start=0.2,
        sigma_min=0.05,
        alpha=0.0,
    )
    save_path = "./../data/headless_bugtrap_cem"
    

elif args.algorithm == "mtp":
    print("Running MTP")
    seed = 0
    ctrl = MTP(
            task,
            num_samples=128,
            M=4,
            N=32,
            sigma_start=0.15, # 0.2
            temperature=0.01,
            num_elites=8, # 2
            beta=0.55, # 0.3
            alpha=0.0,
            interpolation='akima',
            num_randomizations=1,
            seed=seed,
        )
    save_path = "./../data/headless_bugtrap_mtp"

elif args.algorithm == "anmtp":
    print("Running Annealed MTP")
    seed = 0
    ctrl = AnMTP(
            task,
            num_samples=32,
            M=2,
            N=32,
            sigma_min=0.2,
            num_elites=2,
            beta=0.85,
            alpha=0.25,
            interpolation='bspline',
            num_randomizations=1,
            seed=seed,
        )
    save_path = "./../data/headless_bugtrap_anmtp"
    
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model, mj_data = task.reset()

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=25,
    show_traces=True,
    trace_width=0.25,
    max_traces=25,
    record_video=True,
)
