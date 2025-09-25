"""
Example of Local inverse kinematics for a Kuka iiwa robot.

This example demonstrates how to use the mjinx library to solve local inverse kinematics
for a Kuka iiwa robot. It shows how to set up the problem, add tasks and barriers,
and visualize the results using MuJoCo's viewer.
"""

import jax
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
import numpy as np
from mujoco import viewer
from time import perf_counter
from collections import defaultdict

from mjinx.components.barriers import JointBarrier, PositionBarrier, SelfCollisionBarrier
from mjinx.components.tasks import FrameTask
from mjinx.configuration import integrate
from mjinx.problem import Problem
from mjinx.solvers import LocalIKSolver

print("=== Initializing ===")


# === Mujoco ===
print("Loading MuJoCo model...")
mj_model = mj.MjModel.from_xml_path("/home/magnus/GitHub/mtp-thesis/hydrax/models/fr3_pushT_pos/scene_mjx.xml")
mj_data = mj.MjData(mj_model)

mjx_model = mjx.put_model(mj_model)

q_min = mj_model.jnt_range[:, 0].copy()
q_max = mj_model.jnt_range[:, 1].copy()


# --- Mujoco visualization ---
print("Setting up visualization...")
# Initialize render window and launch it at the background
mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model)
mj_viewer = viewer.launch_passive(
    mj_model,
    mj_data,
    show_left_ui=False,
    show_right_ui=False,
)

# Initialize a sphere marker for end-effector task
renderer.scene.ngeom += 1
mj_viewer.user_scn.ngeom = 1
mj.mjv_initGeom(
    mj_viewer.user_scn.geoms[0],
    mj.mjtGeom.mjGEOM_SPHERE,
    0.05 * np.ones(3),
    np.array([0.2, 0.2, 0.2]),
    np.eye(3).flatten(),
    np.array([0.565, 0.933, 0.565, 0.4]),
)

# === Mjinx ===
print("Setting up optimization problem...")
# --- Constructing the problem ---
# Creating problem formulation
problem = Problem(mjx_model, v_min=-100, v_max=100)

# Creating components of interest and adding them to the problem
frame_task = FrameTask("ee_task", cost=1, gain=20, obj_name="ee_frame")

joints_barrier = JointBarrier("jnt_range", gain=10)


problem.add_component(frame_task)

problem.add_component(joints_barrier)


# Compiling the problem upon any parameters update
problem_data = problem.compile()

# Initializing solver and its initial state
print("Initializing solver...")
solver = LocalIKSolver(mjx_model, maxiter=20)

# Initial condition
q = jnp.array([0.0, 0.0, 0.0, 0.0, -np.pi/4, 0.0, -9*np.pi/10, 0.0, 3*np.pi/4, np.pi/4, 0.0, 0.0])

solver_data = solver.init()

# Jit-compiling the key functions for better efficiency
solve_jit = jax.jit(solver.solve)
integrate_jit = jax.jit(integrate, static_argnames=["dt"])

t_warmup = perf_counter()
print("Performing warmup calls...")
# Warmup iterations for JIT compilation
frame_task.target_frame = np.array([0.3, 0.0, 0.05, 0, 0.7071, 0.7071, 0])
problem_data = problem.compile()
opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
q_warmup = integrate_jit(mjx_model, q, velocity=opt_solution.v_opt, dt=1e-2)

t_warmup_duration = perf_counter() - t_warmup
print(f"Warmup completed in {t_warmup_duration:.3f} seconds")

# === Control loop ===
print("\n=== Starting main loop ===")
dt = 1e-2
ts = np.arange(0, 20, dt)

# Performance tracking
solve_times = []
integrate_times = []
n_steps = 0

try:
    for t in ts:
        # Changing desired values
        frame_task.target_frame = np.array([0.3 + 0.2 * jnp.sin(t) ** 2, 0.0, 0.05, 0, 0.7071, 0.7071, 0])

        # After changes, recompiling the model
        problem_data = problem.compile()

        # Solving the instance of the problem
        t1 = perf_counter()
        opt_solution, solver_data = solve_jit(q, solver_data, problem_data)
        t2 = perf_counter()
        solve_times.append(t2 - t1)

        # Integrating
        t1 = perf_counter()
        q = integrate_jit(
            mjx_model,
            q,
            velocity=opt_solution.v_opt,
            dt=dt,
        )
        t2 = perf_counter()
        integrate_times.append(t2 - t1)

        # --- MuJoCo visualization ---
        mj_data.qpos = q
        mj.mj_forward(mj_model, mj_data)
        mj.mjv_initGeom(
            mj_viewer.user_scn.geoms[0],
            mj.mjtGeom.mjGEOM_SPHERE,
            0.05 * np.ones(3),
            np.array(frame_task.target_frame.wxyz_xyz[-3:], dtype=np.float64),
            np.eye(3).flatten(),
            np.array([0.565, 0.933, 0.565, 0.4]),
        )

        # Run the forward dynamics to reflec
        # the updated state in the data
        mj.mj_forward(mj_model, mj_data)
        mj_viewer.sync()
        n_steps += 1
except KeyboardInterrupt:
    print("\nSimulation interrupted by user")
except Exception as e:
    print(f"\nError occurred: {e}")
finally:
    renderer.close()

    # Print performance report
    print("\n=== Performance Report ===")
    print(f"Total steps completed: {n_steps}")
    print("\nComputation times per step:")
    if solve_times:
        avg_solve = sum(solve_times) / len(solve_times)
        std_solve = np.std(solve_times)
        print(f"solve          : {avg_solve * 1000:8.3f} ± {std_solve * 1000:8.3f} ms")
    if integrate_times:
        avg_integrate = sum(integrate_times) / len(integrate_times)
        std_integrate = np.std(integrate_times)
        print(f"integrate      : {avg_integrate * 1000:8.3f} ± {std_integrate * 1000:8.3f} ms")

    if solve_times and integrate_times:
        avg_total = sum(t1 + t2 for t1, t2 in zip(solve_times, integrate_times)) / len(solve_times)
        print(f"\nAverage computation time per step: {avg_total * 1000:.3f} ms")
        print(f"Effective computation rate: {1 / avg_total:.1f} Hz")