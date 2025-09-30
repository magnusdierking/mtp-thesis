import time
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

# Load your XML model
# path = "/home/franka/Lab/mtp-thesis/hydrax/models/fr3_pushT_vel/scene_mjx.xml"
path = "/home/magnus/GitHub/mtp-thesis/hydrax/models/fr3_pushT_vel/scene_mjx.xml"
mj_model = mujoco.MjModel.from_xml_path(path)
mx_model = mjx.put_model(mj_model)

def make_batched_data(n_envs: int):
    """Replicate a single mjx.Data into a leading batch dimension."""
    d_single = mjx.make_data(mx_model)
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_envs,) + x.shape),
        d_single
    )

# Single-env step (pure), then vmap it
def step_one(d):
    return mjx.step(mx_model, d)

step_batched = jax.jit(jax.vmap(step_one))

def run_benchmark(n_envs: int, n_steps: int = 100, warmup: int = 10):
    # Create batched data
    d = make_batched_data(n_envs)

    # Example: set different initial qvel[0] per env (like your vmap example)
    init_vel = jnp.linspace(0.0, 1.0, n_envs)
    d = d.replace(qvel=d.qvel.at[:, 0].set(init_vel))

    # Warmup (JIT+steady-state)
    for _ in range(warmup):
        d = step_batched(d)

    # Timed loop
    t0 = time.time()
    trace_path = "/tmp/jax-trace"
    with jax.profiler.trace(trace_path, create_perfetto_link=True):
        for _ in range(n_steps):
            with jax.profiler.StepTraceAnnotation("Step"):
                d = step_batched(d)
    t1 = time.time()

    ms_per_iter = (t1 - t0) * 1000.0 / n_steps
    print(f"{n_envs:>4} envs → {ms_per_iter:7.3f} ms/iter")
    print(f"Perfetto trace written to: {trace_path}.json")
    print("Open it at https://ui.perfetto.dev (or click the printed link above).")

# Test scaling
for n in [256]:
    run_benchmark(n_envs=n)
