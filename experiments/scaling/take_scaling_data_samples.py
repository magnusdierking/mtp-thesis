import argparse

from hydrax.algs import MPPI, MTP, CEM
from hydrax.algs.mtp.an_mtp_opt import AnMTP
from hydrax.utils.files import get_data_path
from hydrax.utils.utils import mujoco_to_scipy_quat, quat_normalize, quat_conj, quat_mul, quat_error_body, quat_to_rotvec


from hydrax.tasks.pusht_franka import PushTFranka
import jax
import jax.numpy as jnp
import numpy as np

from tqdm import tqdm
import os
from chrono import Timer
import os, pickle
from pathlib import Path
import mujoco
from mujoco import mjx

# helper
def differential_IK(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: str = "ee_frame",
    world_site_vel_desired: np.ndarray = np.zeros(2),
    with_null_space: bool = True,
) -> np.ndarray:
    """
    Differential IK for all dofs in the model.
    """
    # Geometric Jacobians at body_id
    jacp = np.zeros((3, model.nv), dtype=np.float64)  # translational
    jacr = np.zeros((3, model.nv), dtype=np.float64)  # rotational

    bodyid = model.body("ee_frame").id
    mujoco.mj_jacBody(model, data, jacp, jacr, bodyid)

    actuator_joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
    actuator_joint_idxs = [model.joint(name).id for name in actuator_joint_names]

    # get current joint positions
    qpos = data.qpos.copy()
    qnow = qpos[jnp.array(actuator_joint_idxs)]
    qhome = np.array([ 0.51199203,  0.1014329,  -0.36340348, -2.9813132,   0.50339095,  3.06692214, -1.92271156])

    # Build jacobian
    J = np.vstack((jacp, jacr))[:,np.array(actuator_joint_idxs)]  # (6, n)
    J_pinv = np.linalg.pinv(J)
    twist = np.concatenate([world_site_vel_desired, np.zeros(4)])

    ee_position_sensor = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_pos"
    )
    sensor_adr_pos = model.sensor_adr[ee_position_sensor]
    ee_pos = data.sensordata[sensor_adr_pos : sensor_adr_pos + 3]

    ee_orientation_sensor = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_quat"
        )
    sensor_adr = model.sensor_adr[ee_orientation_sensor]
    ee_quat = data.sensordata[sensor_adr : sensor_adr + 4]
    ee_quat = np.array(ee_quat)
    goal_quat = np.array([0.0, 0.7071, 0.7071, 0.0])  #([0.0, 0.0, 0.7071, 0.7071])  # Assuming goal orientation is aligned with x-axis
    goal_quat = np.array(goal_quat)
    goal_vec = quat_error_body(goal_quat, ee_quat)                                   # (3,)

    temp = np.concatenate([world_site_vel_desired, np.array([0.035-ee_pos[2]])])
    twist_err = np.concatenate([temp, goal_vec])                 # [ex, ey, ez, ewx, ewy, ewz]
    dq = J_pinv @ twist_err

    if with_null_space:
        N = np.eye(J.shape[1]) - J_pinv @ J
        kp_ori = 10.0
        dq += N @ (kp_ori * (qhome - qnow))

    return dq



def gravity_comp_torque(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    qvel_bak, qacc_bak = data.qvel.copy(), data.qacc.copy()
    data.qvel[:] = 0.0; data.qacc[:] = 0.0
    tau = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, tau)  # tau = g(q) via inverse dynamics solver
    data.qvel[:] = qvel_bak; data.qacc[:] = qacc_bak
    mujoco.mj_forward(model, data)
    return tau



# ------------------------------------------


det_init = {
    "block_pos_x": 0.2,
    "block_pos_y": 0.1,
    "block_angle": np.pi/3,
    "ee_goal_pos": [0.35, 0.0, 0.035]
}

# Define the task (cost and dynamics)
#velocity control
task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=12,
                    sim_steps_per_control_step=4,
                    ctrl_limits={"u_min": jnp.array([-0.4, -0.4]), 
                                "u_max": jnp.array([0.4, 0.4])},
                    trace_sites=["ee_site"],
                    actuation_type='velocity',
                    sampling_space="velocity",
                    det_init=det_init
                )



# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the walker task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross Entropy Method")
subparsers.add_parser("mtp", help="MTP")
subparsers.add_parser("anmtp", help="Annealed MTP")
args = parser.parse_args()


frequency=10
max_step=200
seed = 445 # 36, ... 



# num_samples_sweep = [128, 256, 512, 1024, 2048, 4096]
# num_randomizations_sweep = [1, 2, 4, 8, 16, 32]

num_samples_sweep = [128, 256]
num_randomizations_sweep = [1,2]


path = Path('./scaling_data_samples/')
path.mkdir(parents=True, exist_ok=True)
path = path / f"{args.algorithm}.pkl"

# -1 because initial trime might be not counted
data = np.empty((len(num_samples_sweep), len(num_randomizations_sweep), max_step-1))

for s, num_samples in enumerate(num_samples_sweep):
    for j, num_randomizations in enumerate(num_randomizations_sweep):
        print(f"\nRunning experiment with num_samples: {num_samples}, num_randomizations: {num_randomizations}")

        # Set the controller based on command-line arguments
        if args.algorithm is None: 
            args.algorithm = "mtp"  # Default to MTP
        elif args.algorithm == "mppi":
            print("Running MPPI")
            ctrl = MPPI(
                task,
                num_samples=num_samples,
                noise_level=0.25,
                temperature=0.1,
                num_randomizations=num_randomizations,
                colorize_noise=False,   # !experimental
                alpha=0.1,
                seed=seed
            )
            
        elif args.algorithm == "cem":
            print("Running CEM")
            ctrl = CEM(
                task,
                num_samples=num_samples,
                num_elites=12,
                sigma_start=0.2,
                sigma_min=0.05,
                alpha=0.1,
                num_randomizations=num_randomizations,
            )
            
        elif args.algorithm == "mtp":
            print("Running MTP")
            ctrl = MTP(
                task,
                num_samples=num_samples,
                M=3, # horizon via control points
                N=32, # samples 
                sigma_min=0.1,
                sigma_start=0.2,
                num_elites=12,
                beta=0.35,
                alpha=0.1,
                interpolation='bspline',
                num_randomizations=num_randomizations,
                seed=seed,
            )
            
        elif args.algorithm == "anmtp":
            print("Running AnMTP")
            ctrl = AnMTP(
                    task,
                    num_samples=num_samples,
                    M=3, # horizon via control points
                    N=32, # samples 
                    sigma_min=0.05,
                    sigma_start=0.2,
                    num_elites=24,
                    keep_elites=4,   # !experimental
                    beta = 0.1,
                    beta_lr = 0.1,        # adaptation step size
                    beta_min = 0.05,
                    beta_max = 0.35,
                    alpha=0.1,
                    interpolation='bspline',
                    shift = False,
                    num_randomizations=num_randomizations,
                    seed=seed,
                )


        # test
        mj_model, mj_data = task.reset()
        ctrl.set_seed(seed)

        task_success = False
        replan_period = 1.0 / frequency
        sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
        sim_steps_per_replan = max(sim_steps_per_replan, 1)

        mjx_data = mjx.put_data(mj_model, mj_data)
        mjx_data = mjx_data.replace(
            mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
        )
        policy_params = ctrl.init_params(seed)
        jit_optimize = jax.jit(ctrl.optimize, donate_argnums=(1,))

        # Controller warm-up
        with Timer() as timer:
            jit_optimize = jit_optimize.lower(mjx_data, policy_params).compile()
        warmup_time = timer.elapsed
        print(f"JIT time: {warmup_time:.2f} seconds")
        policy_params, rollouts = jit_optimize(mjx_data, policy_params)

        step = 0
        for step in tqdm(range(max_step)):
            
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            with Timer() as timer:
                policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = timer.elapsed
            if step > 1:
              
                data[s, j, step-1] = plan_time

            for i in range(sim_steps_per_replan):
                t = i * mj_model.opt.timestep
                u = ctrl.get_action(policy_params, t)

              
                #  remap controls if a control mapper is provided
                if ctrl.control_mapper is not None:
                    u = differential_IK(
                            model=mj_model,
                            data=mj_data,
                            # controller.task.ee_body_id,
                            world_site_vel_desired=u,  # Exclude base DOF
                        )
                if ctrl.gravity_compensator:
                    # Gravity compensation for the robot only
                    tau_g = gravity_comp_torque(mj_model, mj_data)
                    # Clear and apply external torques (generalized forces)
                    mj_data.qfrc_applied[:] = 0.0        # clears all user generalized forces
                    mj_data.xfrc_applied[:] = 0.0        # clears any body-space external wrenches
                    mj_data.qfrc_applied[ctrl.task.actuator_joint_idxs] = tau_g[ctrl.task.actuator_joint_idxs]
                # Apply the control to the simulation
                mj_data.ctrl[:] = np.array(u)

                mujoco.mj_step(mj_model, mj_data)

                if np.isnan(u).any():
                    print("NaN detected in control input; stopping current experiment.")
                    break
            step += 1


# save data array as pickle
with open(path, 'wb') as f:
    pickle.dump(data, f)