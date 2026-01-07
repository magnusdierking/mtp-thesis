#!/usr/bin/env python

import os
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.algs import MPPI, MTP, AnMTP
from hydrax.alg_base import SamplingBasedController
from hydrax.tasks.pusht_franka import PushTFranka


def make_jitted_step(ctrl):
    """
    Pure JAX step, same structure as in FR3_PushT.make_jitted_step,
    but without ROS / TF. We only donate policy_params to avoid
    the aliasing issue with mjx_data.
    """

    def step(mjx_data, policy_params,
             lin_t, quat_t,
             robot_q, robot_dq):

        # CPU -> GPU copies (same as your node)
        lin_t = jnp.asarray(lin_t, dtype=jnp.float32)
        quat_t = jnp.asarray(quat_t, dtype=jnp.float32)
        robot_q = jnp.asarray(robot_q, dtype=jnp.float32)
        robot_dq = jnp.asarray(robot_dq, dtype=jnp.float32)

        # Update qpos / qvel (copied from your node)
        new_qpos = mjx_data.qpos.at[0:14].set(jnp.array([
            lin_t[0] - 0.15,
            lin_t[1],
            lin_t[2],
            quat_t[3],
            quat_t[0],
            quat_t[1],
            quat_t[2],
            robot_q[0],
            robot_q[1],
            robot_q[2],
            robot_q[3],
            robot_q[4],
            robot_q[5],
            robot_q[6],
        ], dtype=jnp.float32))

        new_qvel = mjx_data.qvel.at[7:14].set(jnp.array([
            robot_dq[0],
            robot_dq[1],
            robot_dq[2],
            robot_dq[3],
            robot_dq[4],
            robot_dq[5],
            robot_dq[6],
        ], dtype=jnp.float32))

        mjx_data = mjx_data.replace(
            qpos=new_qpos,
            qvel=new_qvel,
        )

        # Controller optimize (same call as in your node)
        new_policy_params, rollouts = ctrl.optimize(mjx_data, policy_params)

        return mjx_data, new_policy_params

    # Only donate policy_params (arg index 1) to avoid double-donation errors
    return jax.jit(step, donate_argnums=(1,))


def break_aliasing(tree):
    """Clone all leaves once to remove internal buffer aliasing."""
    return jax.tree_util.tree_map(lambda x: x + jnp.zeros_like(x), tree)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="algorithm", help="Sampling algorithm (choose one)"
    )
    subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
    subparsers.add_parser("mtp", help="MTP")
    args = parser.parse_args()

    jax.config.update("jax_platform_name", "gpu")

    # --- Build task (same as in your node) ---
    task = PushTFranka(
        ik_type='pinv',
        planning_horizon=12,
        sim_steps_per_control_step=2,
        ctrl_limits={
            "u_min": jnp.array([-0.4, -0.4]),
            "u_max": jnp.array([0.4, 0.4]),
        },
        actuation_type='velocity',
        sampling_space="velocity",
        block_type='free',
    )

    seed = 42

    # --- Build controller (same as your main) ---
    if args.algorithm is None:
        args.algorithm = "mtp"

    if args.algorithm == "mppi":
        print("Running MPPI")
        ctrl = MPPI(
            task,
            num_samples=512,
            noise_level=0.3,
            temperature=0.1,
            num_randomizations=4,
            seed=seed,
        )
    elif args.algorithm == "mtp":
        print("Running MTP")
        ctrl = MTP(
            task,
            num_samples=128,
            M=2,
            N=64,
            num_elites=4,
            beta=0.05,
            alpha=0.01,
            interpolation='bspline',
            num_randomizations=4,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # --- Create mjx_data and policy params (same as node) ---
    mjx_data = mjx.make_data(ctrl.task.model)
    policy_params = ctrl.init_params(seed)

    # One forward (like in your node)
    mjx_data = mjx.forward(ctrl.task.model, mjx_data)

    # Break aliasing once if you want to allow donation later
    mjx_data = break_aliasing(mjx_data)

    # --- Build jitted step ---
    jit_step = make_jitted_step(ctrl)

    # Dummy initial state
    lin0 = jnp.zeros(3, dtype=jnp.float32)
    quat0 = jnp.array([0., 0., 0., 1.], dtype=jnp.float32)
    q0 = jnp.zeros(7, dtype=jnp.float32)
    dq0 = jnp.zeros(7, dtype=jnp.float32)

    print("Compiling jit_step and doing warmup...")
    t0 = time.time()

    # Force compilation
    mjx_data, policy_params = jit_step(mjx_data, policy_params, lin0, quat0, q0, dq0)
    jax.block_until_ready(policy_params)

    # Extra warmup steps
    for _ in range(5):
        mjx_data, policy_params = jit_step(mjx_data, policy_params, lin0, quat0, q0, dq0)
    jax.block_until_ready(policy_params)

    print(f"JIT + warmup time: {time.time() - t0:.3f} s")

    # --- Benchmark loop ---
    num_steps = 200
    dt_list = []

    print(f"Running {num_steps} control steps in a plain Python loop...")
    for i in range(num_steps):
        # You can vary inputs here if you want
        lin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        q = np.zeros(7, dtype=np.float32)
        dq = np.zeros(7, dtype=np.float32)

        t_start = time.time()
        mjx_data, policy_params = jit_step(mjx_data, policy_params, lin, quat, q, dq)
        jax.block_until_ready(policy_params)
        t_end = time.time()

        dt = t_end - t_start
        dt_list.append(dt)

        if (i + 1) % 10 == 0:
            print(f"Step {i+1}/{num_steps}: {dt*1000:.2f} ms")

    dt_arr = np.array(dt_list)
    print("\n=== Benchmark results ===")
    print(f"Mean step time   : {dt_arr.mean()*1000:.2f} ms")
    print(f"Median step time : {np.median(dt_arr)*1000:.2f} ms")
    print(f"95th percentile  : {np.percentile(dt_arr, 95)*1000:.2f} ms")
    print(f"Effective rate   : {1.0/dt_arr.mean():.2f} Hz")


if __name__ == "__main__":
    main()
