import argparse

from hydrax.algs import MPPI, MTP, CEM
from hydrax.algs.mtp.an_mtp_opt import AnMTP
# from hydrax.algs.mtp.an_mtp_dr import AnMTP

from hydrax.utils.files import get_data_path
from hydrax.simulation.deterministic import run_interactive
from hydrax.simulation.deterministic_headless import run_headless_simulation

from hydrax.tasks.pusht_franka import PushTFranka
import jax
import jax.numpy as jnp
import numpy as np

# --------------------------------------------------- #
UPDATE_COV = False
NUM_SAMPLES = 128
NUM_RANDOMIZATIONS = 1
PLANNING_FREQUENCY = 20
PLANNING_HORIZON = 14
SIM_STEPS_PER_CONTROL_STEP = 2
MAX_SPEED = 0.35  # m/s

SIGMA = 0.2
ALPHA = 0.1
TEMPERATURE = 0.1
NUM_ELITES = 24

# ------------------------------------------------- #


for version in ["low", "high", "annealed"]:
    for seed in [40, 41, 42, 43, 44, 45]:
            print(f"Seed: {seed}, Version: {version}")
            
            
            path = get_data_path() / "pushT_sim_extensions" / "annealing"
            if not path.exists():       
                path.mkdir(parents=True, exist_ok=True)
    
            file = f"seed_{seed}_anmtp_{version}.pkl"
            
            if seed == 40:
                det_init = {
                    "block_pos_x": 0.6,
                    "block_pos_y": -0.1,
                    "block_angle": np.pi/2,
                    "ee_goal_pos": [0.45, 0.1, 0.035]
                }
            elif seed == 41:
                det_init = {
                    "block_pos_x": 0.65,
                    "block_pos_y": 0.0,
                    "block_angle": np.pi,
                    "ee_goal_pos": [0.45, 0.1, 0.035]
                }
            elif seed == 42:
                det_init = {
                    "block_pos_x": 0.7,
                    "block_pos_y": 0.05,
                    "block_angle": 5*np.pi/4,
                    "ee_goal_pos": [0.45, 0.1, 0.035]
                }
            elif seed == 43:
                det_init = {
                    "block_pos_x": 0.5,
                    "block_pos_y": 0.15,
                    "block_angle": -np.pi/2,
                    "ee_goal_pos": [0.45, 0.1, 0.035]
                }
            elif seed == 44:
                det_init = {
                    "block_pos_x": 0.4,
                    "block_pos_y": -0.05,
                    "block_angle": -np.pi/4,
                    "ee_goal_pos": [0.45, 0.1, 0.035]
                }
            elif seed == 45:
                det_init = {
                    "block_pos_x": 0.55,
                    "block_pos_y": -0.1,
                    "block_angle": 0*np.pi,
                    "ee_goal_pos": [0.45, 0.1, 0.035]
                }
                
            task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=PLANNING_HORIZON,
                    sim_steps_per_control_step=SIM_STEPS_PER_CONTROL_STEP,
                    ctrl_limits={"u_min": jnp.array([-MAX_SPEED, -MAX_SPEED]), 
                                 "u_max": jnp.array([MAX_SPEED, MAX_SPEED])},
                    trace_sites=[],
                    actuation_type='velocity',
                    sampling_space="velocity",
                    det_init=det_init,
                    block_type = 'free',
                )
  
            if version == "low":
                beta_min = 0.1
                beta_max = 0.1
                beta = 0.1
            elif version == "high":
                beta_min = 0.7
                beta_max = 0.7
                beta = 0.7
            elif version == "annealed":
                beta_min = 0.1
                beta_max = 0.7
                beta = 0.7
                
            ctrl = AnMTP(
                task,
                num_samples=NUM_SAMPLES,
                M=3, # horizon via control points
                N=64, # samples
                sigma_min=SIGMA,
                sigma_max=SIGMA,
                sigma_start=SIGMA,
                num_elites=NUM_ELITES,
                beta = beta,
                beta_lr = 0.1,        # adaptation step size
                beta_min = beta_min,
                beta_max = beta_max,
                alpha=0.0,
                interpolation='bspline',
                shift = True,
                num_randomizations=NUM_RANDOMIZATIONS,
                seed=seed,
                update_cov=UPDATE_COV,
            )
                
            mj_model, mj_data = task.reset(seed=seed)


            # run_interactive(
            #     ctrl,
            #     mj_model,
            #     mj_data,
            #     frequency=20,
            #     show_traces=False,
            #     trace_width=0.25,
            #     max_traces=0,
            #     fixed_camera_id=0,
            #     show_ui=False,
            #     record_video=False,
            #     max_step=400,
            #     seed=seed,
            #     trace_idxs=None,
            #     log_file=path.as_posix(),
            #     )
            
            run_headless_simulation(
                task,
                ctrl,
                frequency=50,
                seeds=[seed],
                max_step=200,
                log_file_prefix=file,
                save_path=path.as_posix(),
                )
