# bench_perfetto.py
import time
import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.pusht_franka import PushTFranka
from hydrax.algs import MTP, AnMTP


def main():
    
    # Define the task (cost and dynamics)
    task = PushTFranka(
        planning_horizon=20,
        sim_steps_per_control_step=2
    )
    seed = 420
    
    controller = MTP(
        task,
        num_samples=64,
        M=10, # horizon via control points
        N=16, # samples 
        sigma_min=0.1,
        sigma_start=0.2,
        num_elites=10,
        beta=0.25,
        alpha=0.1,
        interpolation='bspline',
        num_randomizations=5,
        seed=seed,
    )
    ctrl = AnMTP(
            task,
            num_samples=32,
            M=2,
            N=32,
            sigma_min=0.2,
            num_elites=2,
            beta=0.85,
            alpha=0.25,
            interpolation='bspline',
            num_randomizations=1,
            seed=seed,
        )
    mj_model, mj_data = task.reset(seed=seed)
    
    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(seed)
    
    # ---------- warmup (compiles + runs once) ----------    
    jit_optimize = jax.jit(controller.optimize, donate_argnums=(1,))
    print("Jitting the controller...")
    st = time.time()
    jit_optimize = jit_optimize.lower(mjx_data, policy_params).compile()
    print(f"Time to jit: {time.time() - st:.3f} seconds")

    with jax.profiler.StepTraceAnnotation("Warmup"):
        for _ in range(3):
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    jax.block_until_ready(rollouts) # ensure all device work is finished

    # ---------- perfetto trace of one run ----------
    # The file will contain a clickable link if create_perfetto_link=True
    trace_path = "/tmp/jax-trace"
    n_steps = 10
    # with jax.profiler.trace(trace_path, create_perfetto_link=True):
        # time.sleep(1)  # give the profiler a moment to start
        # run multiple iterations to get a more stable trace
        
        
    t0 = time.time()
    for _ in range(10):
        # with jax.profiler.StepTraceAnnotation("MTP step"):
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    
    t1 = time.time()
            
    s_per_step = (t1 - t0) / n_steps
    ms_per_step = s_per_step * 1000
    print(f"MTP optimize: {ms_per_step:7.3f} ms/step")
    print(f"MTP optimize frequency: {1/s_per_step:7.3f} Hz")
    # print(f"Runtime (steady-state): {t1 - t0:.4f} s")
    # print(f"Perfetto trace written to: {trace_path}.json")
    # print("Open it at https://ui.perfetto.dev (or click the printed link above).")

if __name__ == "__main__":
    main()
