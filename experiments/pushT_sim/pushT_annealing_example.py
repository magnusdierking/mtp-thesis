import argparse
import pickle
import jax.numpy as jnp


from hydrax.algs import MPPI, MTP, CEM
from hydrax.utils.files import get_data_path
from hydrax.algs.mtp.an_mtp_dr import AnMTP
from deterministic_headless_pushT_sweep import run_headless_simulation
from hydrax.tasks.pusht_franka_old import PushTFranka

# --------------------------------------------------- #
UPDATE_COV = False
NUM_SAMPLES = 512
NUM_RANDOMIZATIONS = 1
PLANNING_FREQUENCY = 20
PLANNING_HORIZON = 10
SIM_STEPS_PER_CONTROL_STEP = 2
MAX_SPEED = 0.35  # m/s

SIGMA = 0.2
ALPHA = 0.1
TEMPERATURE = 0.1
NUM_ELITES = 36

# ------------------------------------------------- #

# End effector init close to T

# T init further away with rotation


task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=10,
                    sim_steps_per_control_step=2,
                    ctrl_limits={"u_min": jnp.array([-MAX_SPEED, -MAX_SPEED]), 
                                 "u_max": jnp.array([MAX_SPEED, MAX_SPEED])},
                    trace_sites=["ee_site"],
                    actuation_type='velocity',
                    sampling_space="velocity",
                    det_init=det_init,
                    block_type = 'free',
                )

path = get_data_path() / "pushT_sim_sweep" / "free"
if not path.exists():       
    path.mkdir(parents=True, exist_ok=True)



for controller in ["mppi", "cem", "mtp"]:
    
    if controller == "mtp":
        ctrl = MTP(
            task,
            num_samples=NUM_SAMPLES,
            M=3, # horizon via control points
            N=32, # samples 
            sigma_min=SIGMA,
            sigma_start=SIGMA,
            num_elites=NUM_ELITES,
            beta=0.35,
            alpha=ALPHA,
            interpolation='bspline',
            num_randomizations=NUM_RANDOMIZATIONS,
            planning_freq=PLANNING_FREQUENCY,
            update_cov=UPDATE_COV,
        )
        print("Running MTP")
    elif controller == "anmtp":
        ctrl = AnMTP(
            task,
            num_samples=NUM_SAMPLES,
            M=3, # horizon via control points
            N=32, # samples 
            sigma_min=0.05,
            sigma_start=0.2,
            num_elites=12,
            keep_elites=4,   # !experimental
            beta = 0.25,
            beta_lr = 0.1,        # adaptation step size
            beta_min = 0.15,
            beta_max = 0.55,
            alpha=0.05,
            interpolation='bspline',
            num_randomizations=1,
        )
        print("Running AnMTP")
        
        print(
            f"Planning with {ctrl.task.planning_horizon} steps "
            f"over a {ctrl.task.planning_horizon * ctrl.task.dt} "
            f"second horizon."
        )

    for seed in [0, 1, 2, 3, 5, 6]:
        
        filename = f"pusht_franka_{controller}_{seed}.pkl"
        run_headless_simulation(
            task,
            ctrl,
            frequency=PLANNING_FREQUENCY,
            seeds=[seed],
            max_step=500,
            log_file_prefix="pusht_franka_" + controller, 
            save_path=path.as_posix(),  
        )
