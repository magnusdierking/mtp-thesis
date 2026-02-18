"""
MJWarp vs MJX Benchmark on Franka Push-T Scene
================================================
Compares simulation speed of mujoco_warp (NVIDIA Warp backend) vs MJX (JAX/XLA backend)
using small sinusoidal control inputs on the Franka velocity-controlled robot.

Usage:
    python experiments/mj_warp_test.py [--n_steps 500] [--envs 8 64 256 1024]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Resolve model path (works whether you run from repo root or experiments/)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_MODEL_DIR = _REPO_ROOT / "hydrax" / "models" / "fr3_pushT_vel"
_SCENE_XML = _MODEL_DIR / "scene_mjx_free.xml"

assert _SCENE_XML.exists(), f"Scene XML not found: {_SCENE_XML}"


def load_mj_model() -> mujoco.MjModel:
    """Load the MuJoCo model once (shared by both backends)."""
    return mujoco.MjModel.from_xml_path(str(_SCENE_XML))


# ═══════════════════════════════════════════════════════════════════════════
#  Control signal generation
# ═══════════════════════════════════════════════════════════════════════════


def make_sinusoidal_ctrl_np(
    nu: int,
    n_worlds: int,
    n_steps: int,
    dt: float,
    amplitude: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Generate small sinusoidal control signals.

    Returns shape (n_steps, n_worlds, nu) float32 array.
    Each world gets a different random phase & frequency per actuator so the
    trajectories diverge, making the benchmark more realistic.
    """
    rng = np.random.RandomState(seed)
    freqs = rng.uniform(0.5, 3.0, size=(n_worlds, nu)).astype(np.float32)  # Hz
    phases = rng.uniform(0.0, 2 * np.pi, size=(n_worlds, nu)).astype(np.float32)

    t = np.arange(n_steps, dtype=np.float32)[:, None, None] * dt  # (T, 1, 1)
    ctrl = amplitude * np.sin(2 * np.pi * freqs[None] * t + phases[None])  # (T, W, nu)
    return ctrl.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  MJX (JAX) benchmark
# ═══════════════════════════════════════════════════════════════════════════


def benchmark_mjx(
    mj_model: mujoco.MjModel, n_worlds: int, n_steps: int, warmup: int = 10
) -> dict:
    """Benchmark MJX with vmap + jit."""
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    nu = mj_model.nu
    dt = mj_model.opt.timestep

    # Put model & make batched data
    mx_model = mjx.put_model(mj_model)
    d_single = mjx.make_data(mx_model)
    d_batch = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_worlds,) + x.shape).copy(),
        d_single,
    )

    # Pre-compute controls on device
    ctrl_np = make_sinusoidal_ctrl_np(nu, n_worlds, n_steps + warmup, dt)
    ctrl_jax = jnp.array(ctrl_np)  # (T, W, nu)

    # Batched step
    @jax.jit
    def step_batch(d, ctrl_t):
        d = d.replace(ctrl=ctrl_t)
        return jax.vmap(lambda dd: mjx.step(mx_model, dd))(d)

    # ── warmup (includes JIT compile) ──
    d = d_batch
    for i in range(warmup):
        d = step_batch(d, ctrl_jax[i])
    # Block until done
    jax.block_until_ready(d.qpos)

    # ── timed run ──
    t0 = time.perf_counter()
    for i in range(n_steps):
        d = step_batch(d, ctrl_jax[warmup + i])
    jax.block_until_ready(d.qpos)
    elapsed = time.perf_counter() - t0

    qpos_final = np.array(d.qpos)
    return {
        "elapsed_s": elapsed,
        "ms_per_step": elapsed * 1000.0 / n_steps,
        "steps_per_s": n_steps / elapsed,
        "world_steps_per_s": n_steps * n_worlds / elapsed,
        "qpos_final": qpos_final,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MJWarp (NVIDIA Warp) benchmark
# ═══════════════════════════════════════════════════════════════════════════


def benchmark_mjwarp(
    mj_model: mujoco.MjModel, n_worlds: int, n_steps: int, warmup: int = 10
) -> dict:
    """Benchmark mujoco_warp with nworld parallelism."""
    import mujoco_warp as mjw
    import warp as wp

    nu = mj_model.nu
    dt = mj_model.opt.timestep

    # Put model & make batched data (nworld handles parallelism natively)
    m = mjw.put_model(mj_model)
    d = mjw.make_data(mj_model, nworld=n_worlds)

    # Pre-compute controls
    ctrl_np = make_sinusoidal_ctrl_np(nu, n_worlds, n_steps + warmup, dt)

    # ── warmup ──
    for i in range(warmup):
        wp.copy(d.ctrl, wp.array(ctrl_np[i], dtype=float))
        mjw.step(m, d)
    wp.synchronize()

    # ── timed run ──
    t0 = time.perf_counter()
    for i in range(n_steps):
        wp.copy(d.ctrl, wp.array(ctrl_np[warmup + i], dtype=float))
        mjw.step(m, d)
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    qpos_final = d.qpos.numpy()
    return {
        "elapsed_s": elapsed,
        "ms_per_step": elapsed * 1000.0 / n_steps,
        "steps_per_s": n_steps / elapsed,
        "world_steps_per_s": n_steps * n_worlds / elapsed,
        "qpos_final": qpos_final,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def print_header():
    print("=" * 80)
    print("  MJWarp vs MJX Benchmark  —  Franka Push-T (velocity control)")
    print("=" * 80)


def print_results_table(results: list[dict]):
    """Pretty-print a comparison table."""
    hdr = f"{'Worlds':>8}  {'Backend':>8}  {'ms/step':>10}  {'steps/s':>10}  {'Mworld·step/s':>14}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        mws = r["world_steps_per_s"] / 1e6
        print(
            f"{r['n_worlds']:>8}  {r['backend']:>8}  "
            f"{r['ms_per_step']:>10.3f}  {r['steps_per_s']:>10.1f}  "
            f"{mws:>14.3f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="MJWarp vs MJX benchmark")
    parser.add_argument("--n_steps", type=int, default=500, help="Timed steps per run")
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup steps (incl. JIT)"
    )
    parser.add_argument(
        "--envs",
        type=int,
        nargs="+",
        default=[8, 64, 256, 1024],
        help="List of parallel-world counts to benchmark",
    )
    parser.add_argument("--skip_mjx", action="store_true", help="Skip MJX benchmark")
    parser.add_argument(
        "--skip_warp", action="store_true", help="Skip MJWarp benchmark"
    )
    args = parser.parse_args()

    print_header()

    mj_model = load_mj_model()
    print(f"Model: {_SCENE_XML.name}")
    print(f"  nq={mj_model.nq}  nv={mj_model.nv}  nu={mj_model.nu}")
    print(f"  timestep={mj_model.opt.timestep}s")
    print(f"  n_steps={args.n_steps}  warmup={args.warmup}")
    print(f"  env counts: {args.envs}")

    all_results = []

    for n_worlds in args.envs:
        print(f"\n{'─' * 60}")
        print(f"  {n_worlds} parallel worlds")
        print(f"{'─' * 60}")

        # ── MJWarp ──
        if not args.skip_warp:
            try:
                print(
                    f"  [MJWarp]  running {args.n_steps} steps … ", end="", flush=True
                )
                res_warp = benchmark_mjwarp(
                    mj_model, n_worlds, args.n_steps, args.warmup
                )
                print(
                    f"{res_warp['ms_per_step']:.3f} ms/step  "
                    f"({res_warp['world_steps_per_s']:.0f} world·step/s)"
                )
                all_results.append(
                    {**res_warp, "backend": "MJWarp", "n_worlds": n_worlds}
                )
            except Exception as e:
                print(f"FAILED: {e}")

        # ── MJX ──
        if not args.skip_mjx:
            try:
                print(f"  [MJX]    running {args.n_steps} steps … ", end="", flush=True)
                res_mjx = benchmark_mjx(mj_model, n_worlds, args.n_steps, args.warmup)
                print(
                    f"{res_mjx['ms_per_step']:.3f} ms/step  "
                    f"({res_mjx['world_steps_per_s']:.0f} world·step/s)"
                )
                all_results.append({**res_mjx, "backend": "MJX", "n_worlds": n_worlds})
            except Exception as e:
                print(f"FAILED: {e}")

        # ── Compare final states if both ran ──
        if not args.skip_warp and not args.skip_mjx:
            try:
                warp_q = all_results[-2]["qpos_final"]
                mjx_q = all_results[-1]["qpos_final"]
                # They may have different shapes: warp is (nworld, nq), mjx is (nworld, nq)
                if warp_q.shape == mjx_q.shape:
                    max_diff = np.max(np.abs(warp_q - mjx_q))
                    mean_diff = np.mean(np.abs(warp_q - mjx_q))
                    print(f"  [Δqpos]  max={max_diff:.6e}  mean={mean_diff:.6e}")
                else:
                    print(
                        f"  [Δqpos]  shapes differ: warp={warp_q.shape} mjx={mjx_q.shape}"
                    )
            except Exception:
                pass

    # ── Summary table ──
    if all_results:
        print("\n" + "=" * 80)
        print("  Summary")
        print("=" * 80)
        print_results_table(all_results)

        # Speedup per env count
        if not args.skip_warp and not args.skip_mjx:
            print("Speedup (MJWarp / MJX):")
            for n in args.envs:
                warp_res = [
                    r
                    for r in all_results
                    if r["backend"] == "MJWarp" and r["n_worlds"] == n
                ]
                mjx_res = [
                    r
                    for r in all_results
                    if r["backend"] == "MJX" and r["n_worlds"] == n
                ]
                if warp_res and mjx_res:
                    speedup = mjx_res[0]["ms_per_step"] / warp_res[0]["ms_per_step"]
                    faster = "MJWarp" if speedup > 1 else "MJX"
                    ratio = speedup if speedup > 1 else 1.0 / speedup
                    print(f"  {n:>6} worlds: {faster} is {ratio:.2f}x faster")
            print()


if __name__ == "__main__":
    main()
