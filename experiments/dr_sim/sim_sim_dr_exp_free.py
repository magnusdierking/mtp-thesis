import argparse
import sys

from hydrax.algs import MPPI, MTP, CEM
# from hydrax.algs.mtp.an_mtp_opt import AnMTP
from an_mtp_dr import AnMTP

from hydrax.utils.files import get_data_path
from deterministic_dr import run_interactive

from hydrax.tasks.pusht_franka import PushTFranka
import jax.numpy as jnp
import numpy as np

from hydrax.risk import RiskStrategy, ExpectedCost, AverageCost, ValueAtRisk, ConditionalValueAtRisk,InverseValueAtRisk, InverseConditionalValueAtRisk
from domain_adaptation import UniformDomainRandomization

"""
Run an interactive simulation of the push-T task with predictive sampling.
"""


NUM_SAMPLES = 256     
NUM_RANDOMIZATIONS = 24   
MAX_SPEED = 0.35  # m/s

seed = 42
update_cov = False
sigma_max = 0.35
sigma_min = 0.15
sigma_start = 0.2
# joint
# det_init = {
#     "block_pos_x": 0.1,
#     "block_pos_y": 0.15,
#     "block_angle": np.pi/3,
#     "ee_goal_pos": [0.5, -0.0, 0.035]
# }
#free hard
# det_init = {
#     "block_pos_x": 0.4,
#     "block_pos_y": 0.15,
#     "block_angle": np.pi/3,
#     "ee_goal_pos": [0.4, 0.0, 0.035]
# }
# free easy
det_init = {
    "block_pos_x": 0.45,
    "block_pos_y": 0.15,
    "block_angle": np.pi/4,
    "ee_goal_pos": [0.35, 0.25, 0.035]
}


# Define the task (cost and dynamics)
#velocity control
task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=20,
                    sim_steps_per_control_step=2,
                    ctrl_limits={"u_min": jnp.array([-MAX_SPEED, -MAX_SPEED]), 
                                 "u_max": jnp.array([MAX_SPEED, MAX_SPEED])},
                    actuation_type='velocity',
                    sampling_space="velocity",
                    det_init=det_init,
                    block_type = 'free',
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
algorithm_subparsers.add_parser("anmtp")

# Domain randomization argument (normal argument, not subparser)
parser.add_argument(
    "--dr",
    choices=["uniform", "evolutionary", "bayesian"],
    help="Domain randomization strategy"
)
parser.add_argument(
    "--risk",
    choices=["average", "expectation", "var", "cvar", "ivar", "icvar"],
    help="risk aggregation method for domain randomization"
)

args = parser.parse_args()
print(args)

risk_alpha = 0.25  # for (C)VaR
if args.risk is None or args.risk == "average": 
    args.risk = "average"  # Default to MTP
    aggregation = AverageCost()
elif args.risk == "expectation":
    # initialize with uniform weights
    aggregation = ExpectedCost(jnp.ones((NUM_RANDOMIZATIONS,), dtype=jnp.float32) / NUM_RANDOMIZATIONS)
elif args.risk == "var":
    aggregation = ValueAtRisk(alpha=risk_alpha)
elif args.risk == "cvar":
    aggregation = ConditionalValueAtRisk(alpha=risk_alpha)
elif args.risk == "ivar":
    aggregation = InverseValueAtRisk(alpha=risk_alpha)
elif args.risk == "icvar":
    aggregation = InverseConditionalValueAtRisk(alpha=risk_alpha)



# Set the controller based on command-line arguments
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
        colorize_noise=False,   # !experimental
        alpha=0.1,
        seed=seed,
        update_cov=update_cov,
    )
    error_log = "./../data/error_log_pushT/mppi_{seed}.npy".format(seed=seed)
    
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
        M=3, # horizon via control points
        N=64, # samples
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_start=sigma_start,
        num_elites=12,
        beta=0.25,
        alpha=0.1,
        interpolation='bspline',
        num_randomizations=NUM_RANDOMIZATIONS,
        seed=seed,
        update_cov=update_cov,
    )
    error_log = "./../data/error_log_pushT/mtp_{seed}.npy".format(seed=seed)
    
elif args.algorithm == "anmtp":
    print("Running AnMTP")
    ctrl = AnMTP(
            task,
            num_samples=NUM_SAMPLES,
            M=3, # horizon via control points
            N=64, # samples
            planning_frequency=5,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_start=sigma_start,
            num_elites=12,
            keep_elites=1,   # !experimental
            beta = 0.3,
            beta_lr = 0.1,        # adaptation step size
            beta_min = 0.25,
            beta_max = 0.35,
            alpha=0.1,
            interpolation='bspline',
            num_randomizations=NUM_RANDOMIZATIONS,
            risk_strategy=aggregation,
            seed=seed,
        )
    error_log = "./../data/error_log_pushT/anmtp_{seed}.npy".format(seed=seed)
    
# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)

print("Using Uniform Domain Randomization.")
dr_strategy = UniformDomainRandomization(
    seed=seed,
    task=task,
    controller=ctrl,
    randomized_bodies={"bottom": {"field": "geom_friction", "min": [0.0001], "max": [10], "internal_idx": [0]},
                        "top": {"field": "geom_friction", "min": [0.0001], "max": [10], "internal_idx": [0]},
                        "ground": {"field": "geom_friction", "min": [0.0001], "max": [10], "internal_idx": [0]},
                        
    },
    randomized_joints = {
        # "T_x": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
        # "T_y": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
        # "T_z": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
        # "dof_frictionloss": (0.0, 1.0, ["T_x", "T_y"]),  # randomize frictionloss of T
    },
    num_randomizations=NUM_RANDOMIZATIONS,
)



path = get_data_path() / "dr_sim"  
if not path.exists():       
    path.mkdir(parents=True, exist_ok=True)
path = path / f"seed_{seed}_{args.algorithm}_{args.dr}_{args.risk}"


ctrl.init_randomization_model(dr_strategy.get_current_randomizations())

new_randomizations = dr_strategy.get_current_randomizations()
print("New randomizations:", new_randomizations["geom_friction"][:,[4,5]])

# sys.exit()



num_traces = 1
incr = NUM_SAMPLES // num_traces
trace_idxs = [i * incr for i in range(num_traces)]
print("Tracing indices:", trace_idxs)

run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=5,
    show_traces=True,
    trace_width=0.55,
    max_traces=num_traces,
    fixed_camera_id=0,
    show_ui=True,
    record_video=False,
    max_step=200,
    seed=seed,
    # log_file=path.as_posix(),
    dr_strategy = dr_strategy,
    trace_idxs=trace_idxs,
    )
