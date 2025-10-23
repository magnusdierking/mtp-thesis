import argparse
import pickle
import jax.numpy as jnp


from hydrax.algs import MPPI, MTP, CEM
from hydrax.utils.files import get_data_path
from hydrax.algs.mtp.an_mtp_dr import AnMTP
from deterministic_headless_pushT_sweep import run_headless_simulation
from hydrax.tasks.pusht_franka import PushTFranka

"""
Run an interactive simulation of the push-T task with predictive sampling.
"""

# Define the task (cost and dynamics)
task = PushTFranka(ik_type = 'pinv',
                   planning_horizon=16,
                   sim_steps_per_control_step=4,
                   ctrl_limits={"u_min": jnp.array([-0.45, -0.45]), "u_max": jnp.array([0.45, 0.45])},
                   trace_sites=["ee_site", "T_1", "T_2"],
                   actuation_type='velocity',
                   )

data = {}
path = get_data_path() / "pushT_sim"
path.mkdir(parents=True, exist_ok=True)

for controller in ["mppi", "cem", "mtp", "anmtp"]:
    
    if controller == "mppi":
        ctrl = MPPI(
            task,
            num_samples=128,
            noise_level=0.3,
            temperature=0.1,
            num_randomizations=1,
            colorize_noise=False,   # !experimental
            alpha=0.1,
            seed=seed
        )
        print("Running MPPI")
    elif controller == "cem":
        ctrl = CEM(
            task,
            num_samples=128,
            num_elites=12,
            sigma_start=0.2,
            sigma_min=0.05,
            alpha=0.1,
        )
        print("Running CEM")
    elif controller == "mtp":
        ctrl = MTP(
            task,
            num_samples=128,
            M=3, # horizon via control points
            N=32, # samples 
            sigma_min=0.1,
            sigma_start=0.2,
            num_elites=12,
            beta=0.35,
            alpha=0.1,
            interpolation='bspline',
            num_randomizations=1,
            seed=seed,
        )
        print("Running MTP")
    elif controller == "anmtp":
        ctrl = AnMTP(
            task,
            num_samples=128,
            M=3, # horizon via control points
            N=32, # samples 
            sigma_min=0.05,
            sigma_start=0.2,
            num_elites=12,
            keep_elites=4,   # !experimental
            beta = 0.25,
            beta_lr = 0.1,        # adaptation step size
            beta_min = 0.05,
            beta_max = 0.35,
            alpha=0.05,
            interpolation='bspline',
            num_randomizations=1,
            seed=seed,
        )
        print("Running AnMTP")

    for seed in [0, 1, 2, 3, 4]:
        
        mj_model, mj_data = task.reset(seed=seed)
        error_log = f"./error_logs/pusht_franka_{controller}_seed{seed}.log"
        
        
        
        
        
        # TODO should return error log data
        data[controller][str(seed)] = run_headless_simulation(
            task,
            ctrl,
            frequency=50,
            seeds=[seed],
            max_step=500,
            log_file_prefix="pusht_franka_" + controller,   
        )

    
# Save data
with open(path / "sweep_pushT_sim_data.pkl", "wb") as f:
    pickle.dump(data, f)


# run_interactive(
#     ctrl,
#     mj_model,
#     mj_data,
#     frequency=25,
#     show_traces=True,
#     trace_width=0.55,
#     max_traces=16,
#     fixed_camera_id=0,
#     show_ui=True,
#     record_video=False,
#     max_step=500,
#     seed=seed,
#     error_log_path=error_log,
#     )


# run_headless_simulation(
#     task,
#     ctrl,
#     frequency=50,
#     seeds=[seed],
#     max_step=500,
#     log_file_prefix="pusht_franka_" + args.algorithm,
#     save_path="./results"
#     )