# bench_tensorboard.py
import time
import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.pusht_franka import PushTFranka
from hydrax.algs import MTP

def main():
    # --- setup task & controller ---
    task = PushTFranka(trace_sites=[])
    seed = 420
    controller = MTP(
        task, num_samples=128, M=3, N=64, sigma_min=0.1, sigma_start=0.2,
        num_elites=10, beta=0.25, alpha=0.1, interpolation='bspline',
        num_randomizations=2, seed=seed,
    )
    mj_model, mj_data = task.reset(seed=seed)
    mjx_data = mjx.put_data(mj_model, mj_data).replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(seed)

    # --- jit & compile once ---
    jit_optimize = jax.jit(controller.optimize, donate_argnums=(1,))
    print("Compiling…")
    t0 = time.time()
    executable = jit_optimize.lower(mjx_data, policy_params).compile()
    print(f"Compile time: {time.time() - t0:.3f}s")

    # --- warmup run (not profiled) ---
    policy_params, rollouts = executable(mjx_data, policy_params)
    jax.block_until_ready(rollouts)

    # --- start JAX profiler server for TensorBoard to connect ---
    jax.profiler.start_server(9999)  # one per process

    # --- labeled run (you'll capture this from TensorBoard) ---
    with jax.profiler.StepTraceAnnotation("opt_step"):  # shows up by name
        
        for _ in range(200):
            t1 = time.time()
            policy_params, rollouts = executable(mjx_data, policy_params)
            # jax.block_until_ready(rollouts)
            t2 = time.time()
            print(f"Iteration time: {t2 - t1:.4f}s")


    print(f"Runtime (steady-state): {t2 - t1:.4f}s")
    print("Now launch TensorBoard and capture a profile:")
    print("  tensorboard --logdir=/tmp/tb --port=6006")
    print("In TensorBoard: Profile → Capture profile → Address: localhost:9999 → Start")

if __name__ == "__main__":
    main()
