import time
from typing import Sequence
import os
import csv

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController
from hydrax.files import get_root_path
from hydrax.utils.video import VideoRecorder

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""
def inverse_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: jax.Array,
    target_quat: jax.Array = None,
) -> jax.Array:
    """Perform inverse kinematics to find joint positions for a target pose."""
    body_id = model.body("ee_frame").id
    
    # Create writable, float64 NumPy arrays for the Jacobians
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    # Convert target to proper NumPy float64 array
    if target_quat is not None:
        goal = np.concatenate([
            np.asarray(target_pos, dtype=np.float64),
            np.asarray(target_quat, dtype=np.float64)
        ])
    else:
        goal = np.asarray(target_pos, dtype=np.float64)

    # Ensure goal is shape (3,) or (3, 1)
    point = data.xpos[body_id, :3].copy()

    mujoco.mj_jac(model, data, jacp, jacr, point, body_id)

    # Use jacobian transpose control to estimate target q
    q_target = jacp.T @ goal[:3]  # shape: (nv,)

    # Clip to actuator control range
    print(q_target.shape)
    print(model.actuator_ctrlrange.shape)
    q_target = jnp.clip(q_target[3:], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

    return q_target
    
    
def ik_2d(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_vel: jax.Array,
) -> jax.Array:
    """Perform inverse kinematics for a 2D task."""
    body_id = model.body("ee_frame").id
    dq_curr = data.qvel 
            
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    point = data.xpos[body_id, :3].copy()
    mujoco.mj_jac(model, data, jacp, None, point, body_id)

    # get ee z pos and vel
    z_pos = data.xpos[body_id, 2]
    z_vel = jacp[2, 3:] @ dq_curr[3:]  # Exclude base DOF
    
    Kp_z = 5.0
    Kd_z = 1.0
    error = (z_pos - 0.08)
    jax.lax.cond(error > 0, lambda x: x / 3, lambda x: x, operand=Kp_z)
    z_vel_feedback = - Kp_z * error - Kd_z * z_vel
    goal = np.array([target_vel[0], target_vel[1], z_vel_feedback], dtype=np.float64) # this is for 2d case

    J_xyz = jacp[:, 3:]  # Exclude base DOF
    lam = 1e-3
    J_xyz_damped_pinv = J_xyz.T @ jnp.linalg.inv(J_xyz @ J_xyz.T + lam * jnp.eye(J_xyz.shape[0]))
    
    dq = J_xyz_damped_pinv @ goal
    dq = jnp.clip(dq, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    return dq



def run_interactive(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    delay_ctrl_start: int = 0,
    max_step: float = 1e4,
    log_file: str = None,
    record_video: bool = False,
    show_ui: bool = True,
    seed: int = 0,
) -> None:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        fixed_camera_id: The camera ID to use for the fixed camera view.
        show_traces: Whether to show traces for the site positions.
        max_traces: The maximum number of traces to show at once.
        trace_width: The width of the trace lines (in pixels).
        trace_color: The RGBA color of the trace lines.
        reference: The reference trajectory (qs) to visualize.
        reference_fps: The frame rate of the reference trajectory.
        delay_ctrl_start: The number of simulation steps to delay the controller
                          start by.
        log_file: The directory to save the logs to. If None, no logs are saved.
        max_step: The maximum number of simulation steps to run before timeout.
    """
    logs = []

    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.task.planning_horizon} steps "
        f"over a {controller.task.planning_horizon * controller.task.dt} "
        f"second horizon."
    )

    # Figure out how many sim steps to run before replanning
    task_success = False
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Sim Step/Replan {sim_steps_per_replan} steps, "
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0/mj_model.opt.timestep} Hz"
    )

    # Initialize the controller
    mjx_data = mjx.put_data(mj_model, mj_data)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(seed)
    jit_optimize = jax.jit(controller.optimize, donate_argnums=(1,))
    #jit_optimize = controller.optimize

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        assert reference.shape[1] == mj_model.nq
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)

        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC  # only show dynamic bodies
    
    # Initialize video recording if enabled
    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=(get_root_path() / "recordings").as_posix(),
            width=width,
            height=height,
            fps=actual_frequency,
        )
        # Ensure model visual offscreen buffer is compatible with video recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=show_ui, show_right_ui=show_ui) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(controller.task.trace_site_ids)
            for i in range(
                num_trace_sites * num_traces * controller.task.planning_horizon
            ):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(trace_color),
                )
                viewer.user_scn.ngeom += 1
        
        # Add geometry for the ghost reference
        if reference is not None:
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )

        step = 0
        while viewer.is_running():
            step += 1
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            # Do a replanning step
            plan_start = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = time.time() - plan_start

            # Visualize the rollouts
            if show_traces:
                ii = 0
                for k in range(num_trace_sites):
                    for i in range(num_traces):
                        for j in range(controller.task.planning_horizon):
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[ii],
                                mujoco.mjtGeom.mjGEOM_LINE,
                                trace_width,
                                rollouts.trace_sites[i, j, k],
                                rollouts.trace_sites[i, j + 1, k],
                            )
                            ii += 1

            # Update the ghost reference
            if reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model,
                    ref_data,
                    vopt,
                    pert,
                    viewer.cam,
                    catmask,
                    viewer.user_scn,
                )

            # Step the simulation
            for i in range(sim_steps_per_replan):
                t = i * mj_model.opt.timestep
                u = controller.get_action(policy_params, t)
                # if any u is nan, stop 
                print(f"Control action shape: {u.shape}")
                print(f"Control action: {u}")

                if delay_ctrl_start > 0:
                    delay_ctrl_start -= 1
                    # print(f"Delaying controller start for {delay_ctrl_start} steps")
                else:
                    # remap controls if a control mapper is provided
                    if controller.control_mapper is not None:
                        # use ik functiojn to remap controls
                        # u = inverse_kinematics(
                        #     mj_model,
                        #     mj_data,
                        #     u,  # Exclude base DOF
                        # )
                        u = ik_2d(
                            mj_model,
                            mj_data,
                            u,  # Exclude base DOF
                        )
                        print(f"Remapped control action: {u}")
                    # Apply the control to the simulation
                    mj_data.ctrl[:] = np.array(u)
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
            
            # Capture frame if recording
            if record_video and recorder.is_recording:
                renderer.update_scene(mj_data, viewer.cam)
                frame = renderer.render()
                recorder.add_frame(frame.tobytes())
                
            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )
            # Check for task success
            task_success |= controller.task.success(mj_data)

            # Log data for the current step
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
                print("Control action is NaN, stopping simulation.")
                break
            
            # Stop if max time is exceeded
            if step > max_step:
                print("\nSimulation timed out.")
                break

    # Preserve the last printout
    print("")
    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()

    # Save logs to a CSV file if specified
    if log_file:
        with open(log_file, "w", newline="") as csvfile:
            fieldnames = ["step", "sim_time", "plan_time", "qpos", "qvel", "control", "running_cost", "state_cost", "success"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for log in logs:
                writer.writerow(log)
        print(f"\nLogs saved to {log_file}")