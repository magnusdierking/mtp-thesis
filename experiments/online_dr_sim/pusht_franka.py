import argparse

from hydrax.algs import MPPI, MTP, CEM
# from hydrax.algs.mtp.an_mtp_opt import AnMTP
from an_mtp_dr import AnMTP

from hydrax.utils.files import get_data_path
from deterministic_dr_new import run_interactive

from pusht_franka_dr import PushTFranka
import jax
import jax.numpy as jnp
import numpy as np

from hydrax.risk import RiskStrategy, ExpectedCost, AverageCost, ValueAtRisk, ConditionalValueAtRisk,InverseValueAtRisk, InverseConditionalValueAtRisk
from domain_adaptation import BayesianDomainRandomization, EvolutionaryDomainRandomization, UniformDomainRandomization

"""
Run an interactive simulation of the push-T task with predictive sampling.
"""


num_samples = 256
num_randomizations = 16#30

# very hard cna result in failure
# online_dr = True
# seed = 445
# update_cov = False
# aggregation = "expectation"
# sigma_max = 0.75
# sigma_min = 0.05
# sigma_start = 0.2
# det_init = {
#     "block_pos_x": -0.1,
#     "block_pos_y": 0.1,
#     "block_angle": 3*np.pi/4,
#     "ee_goal_pos": [0.35, 0.0, 0.035]
# }


seed = 42
online_dr = True
update_cov = False
sigma_max = 0.75
sigma_min = 0.05
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
                    planning_horizon=14,
                    sim_steps_per_control_step=2,
                    ctrl_limits={"u_min": jnp.array([-0.4, -0.4]), 
                                 "u_max": jnp.array([0.4, 0.4])},
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
    aggregation = ExpectedCost(jnp.ones((num_randomizations,), dtype=jnp.float32) / num_randomizations)
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
        num_samples=num_samples,
        noise_level=0.2,
        temperature=0.1,
        num_randomizations=num_randomizations,
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
        num_samples=num_samples,
        num_elites=12,
        sigma_start=sigma_start,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        alpha=0.1,
        num_randomizations=num_randomizations,
        seed=seed,
        update_cov=update_cov,
    )
    error_log = "./../data/error_log_pushT/cem_{seed}.npy".format(seed=seed)
    
elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=num_samples,
        M=3, # horizon via control points
        N=64, # samples
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_start=sigma_start,
        num_elites=12,
        beta=0.25,
        alpha=0.1,
        interpolation='bspline',
        num_randomizations=num_randomizations,
        seed=seed,
        update_cov=update_cov,
    )
    error_log = "./../data/error_log_pushT/mtp_{seed}.npy".format(seed=seed)
    
elif args.algorithm == "anmtp":
    print("Running AnMTP")
    ctrl = AnMTP(
            task,
            num_samples=num_samples,
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
            num_randomizations=num_randomizations,
            risk_strategy=aggregation,
            seed=seed,
        )
    error_log = "./../data/error_log_pushT/anmtp_{seed}.npy".format(seed=seed)
    
# Define the model used for simulation
mj_model, mj_data = task.reset(seed=seed)

if args.dr is None:
    online_dr = False
    print("No online domain randomization.")
    dr_strategy = None
elif args.dr == "uniform":
    print("Using Uniform Domain Randomization.")
    dr_strategy = UniformDomainRandomization(
        seed=seed,
        task=task,
        controller=ctrl,
        randomized_bodies={"bottom": {"field": "geom_friction", "min": [0.0001], "max": [0.8], "internal_idx": [2]},
                           "top": {"field": "geom_friction", "min": [0.0001], "max": [0.8], "internal_idx": [2]}
        },
        randomized_joints = {
            # "T_x": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
            # "T_y": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
            # "T_z": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
            # "dof_frictionloss": (0.0, 1.0, ["T_x", "T_y"]),  # randomize frictionloss of T
        },
        num_randomizations=num_randomizations,
    )
elif args.dr == "evolutionary":
    print("Using Evolutionary Domain Randomization.")
    dr_strategy = EvolutionaryDomainRandomization(
        seed=seed,
        task=task,
        controller=ctrl,
        randomized_bodies={"bottom": {"field": "geom_friction", "min": [0.0001], "max": [0.8], "internal_idx": [2]},
                           "top": {"field": "geom_friction", "min": [0.0001], "max": [0.8], "internal_idx": [2]}
        },
        randomized_joints = {
            # "T_x": {"field": "dof_damping", "min": 0.01, "max": 3.0},
            # "T_x": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
            # "T_y": {"field": "dof_damping", "min": 0.01, "max": 3.0},
            # "T_z": {"field": "dof_damping", "min": 0.01, "max": 3.0},
            # "T_y": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
            # "dof_frictionloss": (0.0, 1.0, ["T_x", "T_y"]),  # randomize frictionloss of T
        },
        num_randomizations=num_randomizations,
        elite_fraction=0.2,
    )
elif args.dr == "bayesian":
    print("Using Bayesian Domain Randomization.")
    dr_strategy = BayesianDomainRandomization(
        seed=seed,
        task=task,
        controller=ctrl,
        randomized_bodies={},#{"block": {"field": "body_mass", "min": 0.1, "max": 1.75}},
            # "body_mass": (0.1, 1.75, task.T_bid),  # randomize mass of the block
            # "geom_friction": (jnp.array([0.5, 1e-03, 0.5e-04]), jnp.array([1.5, 10e-03, 2e-04]), task.T_bid),  # friction
        randomized_joints = {
            # "T_x": {"field": "dof_damping", "min": 0.01, "max": 3.0},
            # "T_y": {"field": "dof_damping", "min": 0.01, "max": 3.0},
            # "T_z": {"field": "dof_damping", "min": 0.01, "max": 3.0},
            # "T_x": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
            # "T_y": {"field": "dof_frictionloss", "min": 0.0, "max": 1.0},
            # "dof_frictionloss": (0.0, 1.0, ["T_x", "T_y"]),  # randomize frictionloss of T
        },
        num_randomizations=num_randomizations,
    )



path = get_data_path() / "dr_sim"  
if not path.exists():       
    path.mkdir(parents=True, exist_ok=True)
path = path / f"seed_{seed}_{args.algorithm}_{args.dr}_{args.risk}"

print(dr_strategy.randomized_idxs)
print("+"*10)
# print all shapes
for shape in dr_strategy.get_uniform_randomizations().values():
    print(shape.shape)
# randomizations, geoms, dofs
print(dr_strategy.get_uniform_randomizations()["geom_friction"][:,jnp.array([3, 4]), :])
print("+"*10)

ctrl.init_randomization_model(dr_strategy.get_current_randomizations())

# ----------------
# Test update of randomizations
# # fake signal according to gaussian density around 1
# mu = 1   # mean
# sigma = 0.5   # standard deviation
# new_randomizations = dr_strategy.get_current_randomizations()
# x = np.ones(num_randomizations) 
# fake_signal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
# fake_signal = np.linalg.norm(fake_signal[...,None], axis=1) # lower is better
# fake_signal = np.max(fake_signal) - fake_signal  # invert to make lower better
# # plot
# import matplotlib.pyplot as plt
# plt.plot(x, fake_signal)
# plt.title("Fake performance signal")
# plt.show()


# updated, new_randomizations, weights = dr_strategy.get_updated_randomizations(fake_signal)
# print("New randomizations:", new_randomizations)

# samples = new_randomizations["dof_damping"][:, 0]
# # density plot
# plt.hist(samples, bins=10, density=True)
# plt.title("Damping samples density")
# plt.show()

# exit()


run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=5,
    show_traces=True,
    trace_width=0.55,
    max_traces=6,
    fixed_camera_id=0,
    show_ui=True,
    record_video=False,
    max_step=200,
    seed=seed,
    log_file=path.as_posix(),
    online_dr=online_dr,
    dr_strategy = dr_strategy,
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