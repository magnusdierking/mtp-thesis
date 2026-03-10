import argparse
import sys

# from hydrax.algs import MPPI, CEM, MTP, AnMTP


from mtp_visuals import MTP 
from an_mtp_opt_vis import AnMTP
from mppi_visuals import MPPI
from cem_visual import CEM
from predictive_sampling_visual import PredictiveSampling


from hydrax.risk import WorstCase
from deterministic_beta import run_interactive
# from hydrax.simulation.deterministic import run_interactive

# from hydrax.simulation.deterministic_experiment import run_headless_simulation
from deterministic_state_bins import run_headless_simulation
from hydrax.tasks.bugtrap import BugTrap


"""
Run an interactive simulation of the particle tracking task.

Double click on the green target, then drag it around with [ctrl + right-click].
"""

# Define the task (cost and dynamics)
task = BugTrap(planning_horizon=24, 
               sim_steps_per_control_step=3)
num_samples = 128
sigma = 0.2

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
subparsers.add_parser("ps", help="Predictive Sampling")
args = parser.parse_args()



# Set the controller based on command-line arguments
if args.algorithm is None: 
    args.algorithm = "mtp"  # Default to MTP

elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(task, 
                num_samples=num_samples, 
                noise_level=0.2, 
                temperature=0.01)

elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=num_samples,
        num_elites=8,
        sigma_start=0.2,
        sigma_min=0.15,
        alpha=0.0,
    )
  
elif args.algorithm == "ps":
    print("Running Predictive Sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=num_samples,
        noise_level=0.2,
        num_randomizations=1,
    )

elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
            task,
            num_samples=num_samples,
            M=4,
            N=16,
            temperature=0.01,
            sigma_start=0.2, # 0.2
            sigma_min=0.15,
            sigma_max=0.3,
            num_elites=8, # 2
            beta=0.6, # 0.3
            alpha=0.0,
            interpolation='bspline',
            num_randomizations=1,
        )

else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model, mj_data = task.reset()
save_path = f"./../../data/clean_bugtrap/{args.algorithm}/"

# ----- for inspection -----
# Run the interactive simulation
num_traces = 20
incr = num_samples // num_traces
trace_idxs = [i * incr for i in range(num_traces)]
print("Tracing indices:", trace_idxs)

seed = 10
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=25,
    show_traces=True,
    trace_width=0.25,
    max_traces=25,
    record_video=False,
    seed=seed,
    trace_idxs=trace_idxs,
)
sys.exit()

