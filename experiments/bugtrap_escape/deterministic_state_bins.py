import time
import csv
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from typing import Optional, List
from hydrax.alg_base import SamplingBasedController
from hydrax.task_base import Task
from tqdm import tqdm
import os
from chrono import Timer
import os, pickle
from pathlib import Path
import traceback

def run_headless_simulation(
    task: Task,
    controller: SamplingBasedController,
    frequency: float,
    seeds: List[int],
    delay_ctrl_start: int = 0,
    max_step: int = 10000,
    log_file_prefix: Optional[str] = None,
    save_path: Optional[str] = '.',
    nbr_bins: int = 80,
    data_log_path: str = None,
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
        beta_log = []
        error_log = []
        
        try:
            mj_model, mj_data = task.reset(seed=seed)
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
                policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            warmup_time = timer.elapsed
            print(f"JIT time: {warmup_time:.2f} seconds")
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)

            step = 0
            state_bins = np.zeros((nbr_bins, nbr_bins))  # Initialize state bins for visualization
            for step in tqdm(range(max_step)):
                step += 1
                
                beta = float(policy_params.beta)
                controller.update_beta(beta)
                beta_log.append(beta)
                
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
                        mj_data.ctrl[:] = np.array(u)

                    mujoco.mj_step(mj_model, mj_data)

                    if np.isnan(u).any():
                        print("NaN detected in control input; stopping current experiment.")
                        break
                
                task_success |= controller.task.success(mj_data)
                
                counts = jnp.array(rollouts.state_bins)
                
                state_bins += counts.sum(axis=tuple(range(len(counts.shape) - 2)))

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
            traceback.print_exc()
            continue

        finally:
            # save pickle file for bins
            if log_file_prefix:
                log_dir = Path(save_path)
                log_dir.mkdir(parents=True, exist_ok=True)   # create directory if missing
                log_file = log_dir / f"{log_file_prefix}_seed_{seed}.pkl"
                with open(log_file, "wb") as f:
                    pickle.dump(state_bins, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"State bins saved to {log_file}")
                
                log_file = log_dir / f"{log_file_prefix}_seed_{seed}_beta.pkl"
                with open(log_file, "wb") as f:
                    pickle.dump(beta_log, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Beta log saved to {log_file}")
                
            #########################################################   
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
