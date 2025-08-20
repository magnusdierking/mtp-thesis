import argparse

import evosax
import evosax.algorithms
import mujoco

from mtp.mtp import MTP
from hydrax.algs import MPPI, Evosax, PredictiveSampling
from hydrax.risk import WorstCase
from hydrax.simulation.deterministic import run_interactive
from hydrax.simulation.deterministic_experiment import run_headless_simulation
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
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser(
    "samr", help="Genetic Algorithm with Self-Adaptation Mutation Rate (SAMR)"
)
subparsers.add_parser("de", help="Differential Evolution")
subparsers.add_parser("gld", help="Gradient-Less Descent")
subparsers.add_parser("rs", help="Uniform Random Search")
subparsers.add_parser("mtp", help="MTP")
args = parser.parse_args()

# Set the controller based on command-line arguments
if args.algorithm is None: 
    args.algorithm = "mtp"  # Default to MTP
if args.algorithm == "ps":
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=16,
        noise_level=0.1,
        num_randomizations=10,
        risk_strategy=WorstCase(),
    )

elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(task, num_samples=32, noise_level=0.3, temperature=0.01)
    save_path = "./../data/headless_bugtrap_mppi"

elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(task, evosax.algorithms.Sep_CMA_ES, num_samples=16)

elif args.algorithm == "samr":
    print("Running genetic algorithm with Self-Adaptation Mutation Rate (SAMR)")
    ctrl = Evosax(task, evosax.algorithms.SAMR_GA, num_samples=16)

elif args.algorithm == "de":
    print("Running Differential Evolution (DE)")
    ctrl = Evosax(task, evosax.algorithms.DifferentialEvolution, num_samples=16)

elif args.algorithm == "gld":
    print("Running Gradient-Less Descent (GLD)")
    ctrl = Evosax(task, evosax.algorithms.GradientlessDescent, num_samples=16)

elif args.algorithm == "mtp":
    print("Running MTP")
    seed = 0
    ctrl = MTP(
            task,
            num_samples=32,
            M=3,
            N=32,
            sigma_min=0.1,
            num_elites=3,
            beta=0.25,
            alpha=0.1,
            interpolation='bspline',
            num_randomizations=1,
            seed=seed,
        )
    save_path = "./../data/headless_bugtrap_mtp"
    
elif args.algorithm == "rs":
    print("Running uniform random search")
    # es_params = evosax.strategies.random.EvoParams(
    #     range_min=-1.0,
    #     range_max=1.0,
    # )
    ctrl = Evosax(
        task, evosax.algorithms.RandomSearch, num_samples=16
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=True,
    trace_width=0.25,
    max_traces=25,
    record_video=True,
)

# run headless simulation
# seeds = [0, 1, 2, 3, 4]
# run_headless_simulation(
#     task,
#     ctrl,
#     seeds=seeds,
#     frequency=50,
#     max_step=1000,       
#     log_file_prefix="bugtrap",
#     save_path=save_path,
# )