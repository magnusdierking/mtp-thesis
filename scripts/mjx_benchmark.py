# bench_mjx.py
from functools import partial
import time, argparse, pathlib
import mujoco
from mujoco import mjx
import jax, jax.numpy as jnp
from flax.struct import dataclass
    

def load_mjx(xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    xm = mjx.put_model(m)              # convert to mjx model (DeviceArray-backed)
    xd0 = mjx.make_data(xm)            # initial state
    return xm, xd0, m.opt.timestep, xml_path



def bench(xml, steps=20000, warmup=1000, batch=1, platform=None):
    if platform: jax.config.update("jax_platform_name", platform)

    # load
    xm, xd0, dt, path = load_mjx(xml)
    
    @partial(jax.vmap, in_axes=(None, None, None, 0))
    def rollout_fn(model: mjx.Model, state: mjx.Data, steps: int, rng: jax.Array):
        def policy(step_i):
            u = 0.05 * jnp.sin(0.01*step_i) * jnp.ones(xm.nu)
            return u
        
        def init_state(xd, rng):
            perturb = jax.random.normal(rng, xd.qpos.shape) * 0.01
            return xd.replace(qpos=xd.qpos + perturb)

        def step_fn(xd: mjx.Data, i: int):
            u = policy(i)
            xd = xd.replace(ctrl=u)  # set control input
            xd = mjx.step(xm, xd)
            return xd, None

        def run_batched(xd):
            # init
            xd = init_state(xd, rng)
            # run for a number of steps
            xd, _ = jax.lax.scan(step_fn, xd, jnp.arange(steps))
            return xd
        return run_batched(state)
        
        
    rng = jax.random.key(42)
    warmup_keys = jax.random.split(rng, 1)
    keys = jax.random.split(rng, batch)

    compiled_rollout_fn = jax.jit(rollout_fn, static_argnums=(2,)) # dont trace steps
    # warm start
    t0 = time.perf_counter()
    print('Jax warmup for model ', pathlib.Path(path).name)
    warmup = compiled_rollout_fn(xm, xd0, steps, warmup_keys)
    jax.block_until_ready(warmup.qpos)  # warmup
    print("Warmup done.")
    t1 = time.perf_counter()
    print(f"Warmup took {t1 - t0:.2f}s")
     
     
    t0 = time.perf_counter()
    outB = rollout_fn(xm, xd0, steps, keys)
    jax.block_until_ready(outB.qpos)
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    sim_time = steps * dt * batch
    return {
        "xml": pathlib.Path(path).name,
        "batch": batch,
        "steps": steps,
        "dt": dt,
        "elapsed_s": elapsed,
        "step_time_ms_per_env": 1e3 * elapsed / (steps * batch),
        "total_time_s": elapsed, 
        "sim_speed_x": sim_time / elapsed,  # simulated seconds per wall-second (aggregate over batch)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmls", nargs="+")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1, help="Number of parallel envs")
    ap.add_argument("--platform", choices=["cpu","gpu","tpu"], default=None)
    args = ap.parse_args()

    rows = [bench(x, steps=args.steps, batch=args.batch, platform=args.platform)
            for x in args.xmls]

    keys = ["xml","batch","steps","dt","step_time_ms_per_env","sim_speed_x"]
    head = " | ".join(keys)
    print(head); print("-"*len(head))
    for r in rows:
        print(" | ".join(str(round(r[k],4)) if isinstance(r[k], float) else str(r[k]) for k in keys))

if __name__ == "__main__":
    main()