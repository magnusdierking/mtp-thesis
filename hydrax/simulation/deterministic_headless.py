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

def differential_IK(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: str = "ee_frame",
    world_site_vel_desired: np.ndarray = np.zeros(2),
) -> np.ndarray:
    """
    Differential IK for all dofs in the model.
    """
    # Geometric Jacobians at body_id
    jacp = np.zeros((3, model.nv), dtype=np.float64)  # translational
    jacr = np.zeros((3, model.nv), dtype=np.float64)  # rotational

    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    # Build jacobian
    J = np.vstack((jacp, jacr))  # (6, n)

    # Compute dq with damped pseudo-inverse
    J_pseudo_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
    twist = np.concatenate([world_site_vel_desired, np.zeros(4)])
    dq = J_pseudo_inv @ twist

    return dq


def gravity_comp_torque(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    qvel_bak, qacc_bak = data.qvel.copy(), data.qacc.copy()
    data.qvel[:] = 0.0; data.qacc[:] = 0.0
    tau = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, tau)  # tau = g(q) via inverse dynamics solver
    data.qvel[:] = qvel_bak; data.qacc[:] = qacc_bak
    mujoco.mj_forward(model, data)
    return tau


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
                        #  remap controls if a control mapper is provided
                        if controller.control_mapper is not None:
                            print(f"Original control action: {u}")
                            u = differential_IK(
                                mj_model,
                                mj_data,
                                controller.task.ee_body_id,
                                u,  # Exclude base DOF
                            )
                            print(f"Remapped control action: {u}")
                            
                        # Gravity compensation for the robot only
                        tau_g = gravity_comp_torque(mj_model, mj_data)
                        # Clear and apply external torques (generalized forces)
                        mj_data.qfrc_applied[:] = 0.0        # clears all user generalized forces
                        mj_data.xfrc_applied[:] = 0.0        # clears any body-space external wrenches
                        mj_data.qfrc_applied[controller.task.actuator_joint_idxs] = tau_g[controller.task.actuator_joint_idxs]
                        # Apply the control to the simulation
                        mj_data.ctrl[:] = np.array(u[controller.task.actuator_joint_idxs])

                    mujoco.mj_step(mj_model, mj_data)

                    if np.isnan(u).any():
                        print("NaN detected in control input; stopping current experiment.")
                        break
                
                task_success |= controller.task.success(mj_data)
                                
                logs.append({
                    "step": step,
                    "sim_time": float(mjx_data.time),
                    "plan_time": plan_time,
                    "qpos": np.array(mjx_data.qpos).tolist(),
                    "qvel": np.array(mjx_data.qvel).tolist(),
                    "control": np.array(u).tolist(),
                    "running_cost": jnp.sum(rollouts.costs, axis=1).tolist(),
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
                log_file = os.path.join(save_path, f"{log_file_prefix}_seed_{seed}.csv")
                with open(log_file, "w", newline="") as csvfile:
                    fieldnames = [
                        "step", "sim_time", "plan_time", "qpos", "qvel",
                        "control", "running_cost", "state_cost", "success"
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for log in logs:
                        writer.writerow(log)
                print(f"Logs saved to {log_file}")
