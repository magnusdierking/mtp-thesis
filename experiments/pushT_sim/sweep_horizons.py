from hydrax.algs import MPPI, MTP, CEM, PredictiveSampling

from hydrax.utils.files import get_data_path
from hydrax.simulation.deterministic import run_interactive

from hydrax.tasks.pusht_franka import PushTFranka
import jax.numpy as jnp
import numpy as np


SIM_STEPS_PER_CONTROL_STEP = 2
NUM_ELITES = 32

NUM_SAMPLES = 512
NUM_RANDOMIZATIONS = 1

UPDATE_COV = False
SIGMA = 0.25

MAX_SPEED = 0.35  # m/s


for algo in ["ps", "mppi"]:
    for seed in [10, 42, 445]:
        for planning_horizon in [6, 10, 14, 16, 20]:
            print(f"Algorithm: {algo}, Seed: {seed}, Planning horizon: {planning_horizon}")
            
            
            path = get_data_path() / "pushT_sim_horizons_sweep" / algo
            if not path.exists():       
                path.mkdir(parents=True, exist_ok=True)
    
            path = path / f"seed_{seed}_h_{planning_horizon}.pkl"
            
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
                    planning_horizon=planning_horizon,
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
                    num_elites=NUM_ELITES,
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
                    num_elites=NUM_ELITES,
                    beta=0.25,
                    alpha=0.1,
                    interpolation='bspline',
                    num_randomizations=NUM_RANDOMIZATIONS,
                    seed=seed,
                    update_cov=UPDATE_COV,
                    savgol_filter=True,  # !experimental
                    default_zero_controls=False,
                )
                
            elif algo == "mppi":
                ctrl = MPPI(
                    task,
                    num_samples=NUM_SAMPLES,
                    noise_level=0.3,
                    temperature=0.1,
                    num_randomizations=NUM_RANDOMIZATIONS,
                    colorize_noise=False,   # !experimental
                    alpha=0.1,
                    seed=seed,
                )
            elif algo == "ps":
                ctrl = PredictiveSampling(
                    task,
                    num_samples=NUM_SAMPLES,
                    noise_level=SIGMA,
                    num_randomizations=NUM_RANDOMIZATIONS,
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
