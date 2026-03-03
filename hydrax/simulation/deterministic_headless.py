import time
import csv
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from typing import Optional, List
from hydrax.alg_base_opt import SamplingBasedController
from hydrax.task_base import Task
from tqdm import tqdm
import os
from chrono import Timer
import os, pickle
from pathlib import Path

from hydrax.utils.utils import mujoco_to_scipy_quat, quat_normalize, quat_conj, quat_mul, quat_error_body, quat_to_rotvec


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
    actuator_jids = model.jnt_qposadr[actuator_joint_idxs]
    dof_adr  = model.jnt_dofadr[actuator_joint_idxs]    

    # get current joint positions
    qpos = data.qpos.copy()
    qnow = qpos[jnp.array(actuator_jids)]
    qhome = np.array([ 0.51199203,  0.1014329,  -0.36340348, -2.9813132,   0.50339095,  3.06692214, -1.92271156])

    # Build jacobian
    J = np.vstack((jacp, jacr))[:,np.array(dof_adr)]  # (6, n)
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
    
    # print("End effector translation z error:", 0.035 - ee_pos[2])
    temp = np.concatenate([world_site_vel_desired, np.array([0.045-ee_pos[2]])])
    twist_err = np.concatenate([temp, goal_vec])                 # [ex, ey, ez, ewx, ewy, ewz]
    dq = J_pinv @ twist_err

    if with_null_space:
        N = np.eye(J.shape[1]) - J_pinv @ J
        kp_ori = 10.0
        dq += N @ (kp_ori * (qhome - qnow))

    return dq


def run_headless_simulation(
    task: Task,
    controller: SamplingBasedController,
    frequency: float,
    seeds: List[int],
    delay_ctrl_start: int = 0,
    max_step: int = 10000,
    log_file_prefix: Optional[str] = None,
    save_path: Optional[str] = '.',
) -> None:
    """Run deterministic headless MuJoCo simulations with multiple seeds.

    Args:
        controller: The MPC controller instance.
        mj_model: MuJoCo simulation model.
        mj_data: MuJoCo simulation data object.
        frequency: Control frequency (Hz).
        seeds: List of seeds for multiple experiments.
        delay_ctrl_start: Steps to delay the controller's action.
        max_step: Maximum number of simulation steps.
        log_file_prefix: Prefix path to log simulation data; seed will be appended to filename.
    """
    for seed in seeds:
        print(f"\nRunning experiment with seed: {seed}")
        logs = []
        np.random.seed(seed)

        plan_times = []
        try:
            mj_model, mj_data = task.reset()
            controller.set_seed(seed)

            task_success = False
            replan_period = 1.0 / frequency
            sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
            sim_steps_per_replan = max(sim_steps_per_replan, 1)

            mjx_data = mjx.put_data(mj_model, mj_data)
            mjx_data = mjx_data.replace(
                mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
            )
            policy_params = controller.init_params(seed)
            jit_optimize = jax.jit(controller.optimize, donate_argnums=(1,))

            # Controller warm-up
            with Timer() as timer:
                jit_optimize = jit_optimize.lower(mjx_data, policy_params).compile()
            warmup_time = timer.elapsed
            print(f"JIT time: {warmup_time:.2f} seconds")
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)

            step = 0
            for step in tqdm(range(max_step)):
                step += 1
                
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
                    plan_times.append(plan_time)

                for i in range(sim_steps_per_replan):
                    t = i * mj_model.opt.timestep
                    u = controller.get_action(policy_params, t)

                    if delay_ctrl_start > 0:
                        delay_ctrl_start -= 1
                    else:
                        if controller.control_mapper is not None:
                        # print(f"Original control action: {u}")
                            if controller.task.actuation_type == 'velocity':
                                u = differential_IK(
                                    model=mj_model,
                                    data=mj_data,
                                    # controller.task.ee_body_id,
                                    world_site_vel_desired=u,  # Exclude base DOF
                                )
                        mj_data.ctrl[:] = np.array(u)

                    mujoco.mj_step(mj_model, mj_data)

                    if np.isnan(u).any():
                        print("NaN detected in control input; stopping current experiment.")
                        break
                state_error = controller.task.running_cost(mj_data, u)
                task_success |= controller.task.success(mj_data)
                                
                logs.append({
                    "step": step,
                    "sim_time": float(mjx_data.time),
                    "plan_time": plan_time,
                    "qpos": np.array(mjx_data.qpos).tolist(),
                    "qvel": np.array(mjx_data.qvel).tolist(),
                    "control": np.array(u).tolist(),
                    "state_error": float(state_error),
                    "state_cost": float(rollouts.costs[0, 0]),
                    "success": task_success,
                })

                if np.isnan(u).any():
                    break
            print(f"Task success: {task_success}")

        except Exception as e:
            print(f"Experiment with seed {seed} encountered an error: {e}")
            continue

        finally: 
            plan_times = np.array(plan_times)
            print(f"Iteration time: {np.mean(plan_times)} \\pm {np.std(plan_times)} seconds")
            if log_file_prefix:
                log_file = os.path.join(save_path, f"{log_file_prefix}_seed_{seed}.pkl")
        
                with open(log_file, "wb") as f:
                    pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved to {log_file}")
