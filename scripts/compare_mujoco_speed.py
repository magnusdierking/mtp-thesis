import time
import mujoco
import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.utils.files import get_root_path

IMPL = "warp"
# IMPL = "jax"
N_ENVS = 2048  # Number of parallel environments

def load_model(model_path):
    """Load the MuJoCo model."""
    return mujoco.MjModel.from_xml_path(model_path)

def simulate(model, data, steps):
    """Simulate using vectorized environments."""
    def step_fn(i, data):
        return jax.vmap(lambda d: mjx.step(model, d))(data)
    
    return jax.lax.fori_loop(0, steps, step_fn, data)

def main():
    model_path = "/home/carvalho/Projects/ModelTensorPlanning/mtp-thesis/hydrax/models/fr3_pushT_vel/scene_mjx.xml"
    # model_path = "/home/carvalho/Projects/ModelTensorPlanning/mtp-thesis/hydrax/models/double_cart_pole/scene.xml"
    # model_path = "/home/carvalho/Projects/ModelTensorPlanning/mtp-thesis/mujoco_warp/benchmark/humanoid/humanoid.xml"
    # model_path = "/home/carvalho/Projects/ModelTensorPlanning/mtp-thesis/mujoco_warp/benchmark/franka_emika_panda/scene.xml"
    # model_path = "/home/carvalho/Projects/ModelTensorPlanning/mtp-thesis/hydrax/models/bugtrap/scene.xml"
    
    steps = 100

    # 1. Load model (CPU)
    mj_model = load_model(model_path)

    # 2. Put model on device (GPU/TPU)
    mjx_model = mjx.put_model(mj_model, impl=IMPL)
    
    # Create N_ENVS copies of the initial data using jax.vmap
    # We are calling a function that creates a data structure, and vmap over the output.
    mjx_data = jax.vmap(lambda _: mjx.make_data(mj_model, impl=IMPL, nconmax=N_ENVS * 100, njmax=100))(jnp.arange(N_ENVS))

    # This creates a structure where the batch dimension is the first axis 
    # of every leaf that needs to be batched, AND leaves internal arrays alone.
    
    # Warmup run
    print("Warming up...")
    
    # Re-JIT the simulation function to accept the new batched data structure
    simulate_jit = jax.jit(simulate)
    start_time = time.time()
    mjx_data = simulate_jit(mjx_model, mjx_data, steps)
    end_time = time.time()
    print(f"Warmup completed in {end_time - start_time:.4f} seconds")
    
    # ... (rest of your timing code remains the same)
    print(f"\nRunning {N_ENVS} parallel environments for {steps} steps each:")
    for i in range(10):
        start_time = time.time()
        mjx_data = simulate_jit(mjx_model, mjx_data, steps) # Use the JITted function
        sim_time = time.time() - start_time
        print(f"Run {i+1}: {sim_time:.4f} seconds ({steps * N_ENVS / sim_time:.2f} steps/sec), Frequency {1/(sim_time):.2f} Hz , Avg time per step: {sim_time / (steps * N_ENVS) * 1000:.6f} ms")

if __name__ == "__main__":
    main()