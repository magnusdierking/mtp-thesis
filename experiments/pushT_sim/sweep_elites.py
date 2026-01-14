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
import numpy as np

PLANNING_HORIZON = 10
SIM_STEPS_PER_CONTROL_STEP = 2

NUM_SAMPLES = 512
NUM_RANDOMIZATIONS = 1

UPDATE_COV = False
SIGMA = 0.25

MAX_SPEED = 0.35  # m/s


for algo in ["cem", "mtp"]:
    for seed in [10]:
        for nbr_elites in [8, 16, 32, 64, 128]:
            print(f"Algorithm: {algo}, Seed: {seed}, Num elites: {nbr_elites}")
            
            
            path = get_data_path() / "pushT_sim_elites_sweep" / algo
            if not path.exists():       
                path.mkdir(parents=True, exist_ok=True)
    
            path = path / f"seed_{seed}_n_{nbr_elites}.pkl"
            
            if seed == 10:
                # no rotation error, very hard with local minima
                det_init = {
                    "block_pos_x": 0.45,
                    "block_pos_y": 0.1,
                    "block_angle": -np.pi/2 ,
                    "ee_goal_pos": [0.5, 
                                    0.0, 
                                    0.045]
                }
            elif seed == 42:
                # Head of T towards goal, ee on other side
                det_init = {
                    "block_pos_x": 0.65,
                    "block_pos_y": -0.15,
                    "block_angle": np.pi/6,
                    "ee_goal_pos": [0.45, 0.1, 0.045]
                }
            elif seed == 445:
                # simple setting
                det_init = {
                    "block_pos_x": 0.6,
                    "block_pos_y": 0.1,
                    "block_angle": -np.pi/3,
                    "ee_goal_pos": [0.7, 0.3, 0.045]
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


            if algo == "cem":
                ctrl = CEM(
                    task,
                    num_samples=NUM_SAMPLES,
                    num_elites=nbr_elites,
                    sigma_start=SIGMA,
                    sigma_min=SIGMA,
                    sigma_max=SIGMA,
                    alpha=0.1,
                    num_randomizations=NUM_RANDOMIZATIONS,
                    seed=seed,
                    savgol_filter=True,  # !experimental
                    update_cov=UPDATE_COV,
                )   
            elif algo == "mtp":
                ctrl = MTP(
                    task,
                    num_samples=NUM_SAMPLES,
                    M=3, # horizon via control points
                    N=64, # samples
                    sigma_min=SIGMA,
                    sigma_max=SIGMA,
                    sigma_start=SIGMA,
                    num_elites=nbr_elites,
                    beta=0.25,
                    alpha=0.1,
                    interpolation='bspline',
                    num_randomizations=NUM_RANDOMIZATIONS,
                    seed=seed,
                    update_cov=UPDATE_COV,
                    savgol_filter=True,  # !experimental
                    default_zero_controls=False,
                )
                
            elif algo == "anmtp":
                ctrl = AnMTP(
                    task,
                    num_samples=NUM_SAMPLES,
                    M=3, # horizon via control points
                    N=64, # samples
                    sigma_min=SIGMA,
                    sigma_max=SIGMA,
                    sigma_start=SIGMA,
                    num_elites=nbr_elites,
                    keep_elites=1,   # !experimental
                    beta = 0.25,
                    beta_lr = 0.1,        # adaptation step size
                    beta_min = 0.0,
                    beta_max = 0.35,
                    alpha=0.0,
                    interpolation='bspline',
                    shift = False,
                    num_randomizations=NUM_RANDOMIZATIONS,
                    seed=seed,
                    update_cov=UPDATE_COV,
                )
                
            mj_model, mj_data = task.reset(seed=seed)


            run_interactive(
                ctrl,
                mj_model,
                mj_data,
                frequency=20,
                show_traces=False,
                trace_width=0.25,
                max_traces=0,
                fixed_camera_id=0,
                show_ui=False,
                record_video=False,
                max_step=400,
                seed=seed,
                trace_idxs=None,
                log_file=path.as_posix(),
                )
