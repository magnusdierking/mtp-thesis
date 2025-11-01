import time
from typing import Sequence
import csv
from xml.parsers.expat import model

from hydrax.algs.mtp.beta_scheduler import RatioEMAScheduler
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController
from hydrax.files import get_root_path
from hydrax.utils.video import VideoRecorder

from hydrax.algs.mtp.beta_scheduler import *
import matplotlib.pyplot as plt

from hydrax.utils.utils import mujoco_to_scipy_quat, quat_normalize, quat_conj, quat_mul, quat_error_body, quat_to_rotvec

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""    
# def differential_IK(
#     model: mujoco.MjModel,
#     data: mujoco.MjData,
#     body_id: str = "ee_frame",
#     world_site_vel_desired: np.ndarray = np.zeros(2),
# ) -> np.ndarray:
#     """
#     Differential IK for all dofs in the model, with planar motion priority
#     and roll/pitch anti-tilt damping.
#     """
#     # Geometric Jacobians at body_id
#     jacp = np.zeros((3, model.nv), dtype=np.float64)  # translational
#     jacr = np.zeros((3, model.nv), dtype=np.float64)  # rotational
#     mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

#     # Build 6×nv Jacobian
#     J = np.vstack((jacp, jacr))

#     # --- Roll/pitch anti-tilt (use current angular velocity) ---
#     v_curr = J @ data.qvel                     # [vx, vy, vz, wx, wy, wz]
#     wx, wy = v_curr[3], v_curr[4]
#     k_rp = 2.0                                 # tune ~2..10
#     # Desired 6D twist: vx, vy from input; vz=0; wx/wy damped; wz=0
#     twist = np.array([
#         world_site_vel_desired[0],
#         world_site_vel_desired[1],
#         0.0,
#         -k_rp * wx,                           # kill roll
#         -k_rp * wy,                           # kill pitch
#         0.0                                   # leave yaw free
#     ], dtype=np.float64)

#     # --- Optional row-weighting: prioritize planar translation over orientation ---
#     W = np.diag([1.0, 1.0, 0.0, 0.3, 0.3, 0.0])   # lightweight priority; adjust if needed
#     Jw = J
#     tw = twist

#     # Damped least-squares with the weighted Jacobian
#     lam = 1e-3
#     JwJwT = Jw @ Jw.T
#     J_pseudo_inv = Jw.T @ np.linalg.inv(JwJwT + (lam * lam) * np.eye(6))

#     dq = J_pseudo_inv @ tw
#     return dq

def ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: str = "ee_frame",
    desired_xy: np.ndarray = np.zeros(2),
    num_iters: int = 10,
) -> np.ndarray:

    step_size = 0.05
    actuator_joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
    actuator_joint_idxs = [model.joint(name).id for name in actuator_joint_names]

    for _ in range(num_iters):
        ee_position_sensor = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_pos"
        )
        sensor_adr_pos = model.sensor_adr[ee_position_sensor]
        ee_pos = data.sensordata[sensor_adr_pos : sensor_adr_pos + 3]

        desired_xy_vel = np.clip(desired_xy - ee_pos[:2], -step_size, step_size)

        dq = differential_IK(
            model,
            data,
            body_id=body_id,
            world_site_vel_desired=desired_xy,
        )
        # Apply joint update
        data.qpos[actuator_joint_idxs] += dq
        mujoco.mj_forward(model, data)
    return data.qpos[actuator_joint_idxs]


def differential_IK(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: str = "ee_frame",
    world_site_vel_desired: np.ndarray = np.zeros(2),
    with_null_space: bool = True,
) -> np.ndarray:
    """
    Differential IK for all dofs in the model.
    """
    # Geometric Jacobians at body_id
    jacp = np.zeros((3, model.nv), dtype=np.float64)  # translational
    jacr = np.zeros((3, model.nv), dtype=np.float64)  # rotational

    bodyid = model.body("ee_frame").id
    mujoco.mj_jacBody(model, data, jacp, jacr, bodyid)

    actuator_joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
    actuator_joint_idxs = [model.joint(name).id for name in actuator_joint_names]

    # get current joint positions
    qpos = data.qpos.copy()
    qnow = qpos[jnp.array(actuator_joint_idxs)]
    qhome = np.array([ 0.51199203,  0.1014329,  -0.36340348, -2.9813132,   0.50339095,  3.06692214, -1.92271156])

    # Build jacobian
    J = np.vstack((jacp, jacr))[:,np.array(actuator_joint_idxs)]  # (6, n)
    J_pinv = np.linalg.pinv(J)
    twist = np.concatenate([world_site_vel_desired, np.zeros(4)])

    ee_position_sensor = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_pos"
    )
    sensor_adr_pos = model.sensor_adr[ee_position_sensor]
    ee_pos = data.sensordata[sensor_adr_pos : sensor_adr_pos + 3]

    ee_orientation_sensor = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_quat"
        )
    sensor_adr = model.sensor_adr[ee_orientation_sensor]
    ee_quat = data.sensordata[sensor_adr : sensor_adr + 4]
    ee_quat = np.array(ee_quat)
    goal_quat = np.array([0.0, 0.7071, 0.7071, 0.0])  #([0.0, 0.0, 0.7071, 0.7071])  # Assuming goal orientation is aligned with x-axis
    goal_quat = np.array(goal_quat)
    goal_vec = quat_error_body(goal_quat, ee_quat)                                   # (3,)

    temp = np.concatenate([world_site_vel_desired, np.array([0.035-ee_pos[2]])])
    twist_err = np.concatenate([temp, goal_vec])                 # [ex, ey, ez, ewx, ewy, ewz]
    dq = J_pinv @ twist_err

    if with_null_space:
        N = np.eye(J.shape[1]) - J_pinv @ J
        kp_ori = 10.0
        dq += N @ (kp_ori * (qhome - qnow))

    return dq



def gravity_comp_torque(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    qvel_bak, qacc_bak = data.qvel.copy(), data.qacc.copy()
    data.qvel[:] = 0.0; data.qacc[:] = 0.0
    tau = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, tau)  # tau = g(q) via inverse dynamics solver
    data.qvel[:] = qvel_bak; data.qacc[:] = qacc_bak
    mujoco.mj_forward(model, data)
    return tau



def run_interactive(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.4],
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
    # jit_optimize = jax.jit(controller.optimize, donate_argnums=(0,1))
    #jit_optimize = controller.optimize

    # Warm-up the controller
    # controller.compile_optimize()
    print("Jitting the controller...")
    st = time.time()
    jit_optimize = jit_optimize.lower(mjx_data, policy_params).compile()
    print(f"Time to jit: {time.time() - st:.3f} seconds") 
    
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    
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

        # --- init ---
        step = 0
        #alpha = 0.15          
        #kw = {"beta_min": 0.1, "beta_max": 0.6}
        #sched = RatioEMAScheduler(alpha=alpha, **kw).init(mj_data.qpos)

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

            # ----- adaptive beta (single alpha) -----
            # x = jnp.array(mj_data.qpos)
            # beta = sched.update(x)
            # controller.update_beta(float(beta))
            # print(f"Updated beta to {float(beta):.3f}")
            # -------------------------------------------

            # Do a replanning step
            plan_start = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            # policy_params, rollouts = controller.opt_step(mjx_data, policy_params)
            plan_time = time.time() - plan_start

            if hasattr(controller, 'beta'):
                controller.beta = float(policy_params.beta) # TODO
                
            
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
                                rollouts.trace_sites[0, i, j, k],        # ! 
                                rollouts.trace_sites[0, i, j + 1, k],    # !
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
                # print(f"Control action shape: {u.shape}")
                # print(f"Control action: {u}")

                if delay_ctrl_start > 0:
                    delay_ctrl_start -= 1
                    # print(f"Delaying controller start for {delay_ctrl_start} steps")
                else:
                    # remap controls if a control mapper is provided
                    if controller.control_mapper is not None:
                        # print(f"Original control action: {u}")
                        if controller.task.actuation_type == 'velocity':
                            u = differential_IK(
                                model=mj_model,
                                data=mj_data,
                                # controller.task.ee_body_id,
                                world_site_vel_desired=u,  # Exclude base DOF
                            )
                        elif controller.task.actuation_type == 'position':
                            # u = ik(
                            #     model=mj_model,
                            #     data=mj_data,
                            #     body_id=controller.task.ee_body_id,
                            #     desired_xy=u,
                            # )
                            pass
                        # print(f"Remapped control action: {u}")
                    
                    if controller.gravity_compensator:
                        # Gravity compensation for the robot only
                        tau_g = gravity_comp_torque(mj_model, mj_data)
                        # Clear and apply external torques (generalized forces)
                        mj_data.qfrc_applied[:] = 0.0        # clears all user generalized forces
                        mj_data.xfrc_applied[:] = 0.0        # clears any body-space external wrenches
                        mj_data.qfrc_applied[controller.task.actuator_joint_idxs] = tau_g[controller.task.actuator_joint_idxs]
                        # Apply the control to the simulation
                    # mj_data.ctrl[:] = np.array(mj_data.qpos[np.array(controller.task.actuator_joint_idxs)])
                    mj_data.ctrl[:] = np.array(u)
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
            # data    
            state_error = np.linalg.norm(controller.task._get_position_err(mj_data) 
                        + np.linalg.norm(controller.task._get_orientation_err(mj_data)))
            # only for pusht
            sensor_adr = mj_model.sensor_adr[controller.task.ee_position_sensor]
            ee_pos = mj_data.sensordata[sensor_adr : sensor_adr + 3]

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
            # Check for task success
            task_success |= controller.task.success(mj_data)
            if hasattr(controller, 'beta'):
                print(
                    f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s, sim time: {mj_data.time:.2f}s, success: {task_success:.3f}, beta: {controller.beta:.3f}", 
                    end="\r",
                )
            else:
                print(
                    f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s, sim time: {mj_data.time:.2f}s, success: {task_success:.3f}", 
                    end="\r",
                )

            # Log data for the current step
            logs.append({
                "step": step,
                "sim_time": float(mjx_data.time),
                "plan_time": plan_time,
                "qpos": np.array(mj_data.qpos).tolist(),
                "qvel": np.array(mj_data.qvel).tolist(),
                "ee_pos": np.array(ee_pos).tolist(),
                "control": np.array(u).tolist(),
                "running_cost": jnp.sum(rollouts.costs, axis=1).tolist(),
                "state_error": float(state_error),
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
            fieldnames = ["step", "sim_time", "plan_time", "qpos", "qvel", "ee_pos", "control", "state_error", "running_cost", "state_cost", "success"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for log in logs:
                writer.writerow(log)
        print(f"\nLogs saved to {log_file}")