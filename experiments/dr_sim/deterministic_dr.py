import time
from typing import Sequence
import csv
import pickle
from xml.parsers.expat import model
from pathlib import Path

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

from hydrax.utils.utils import mujoco_to_scipy_quat, quat_normalize, quat_conj, quat_mul, quat_error_body, quat_to_rotvec, mat2quat, se3_left_invariant_metric
from domain_adaptation import AdaptiveDomainRandomizationStrategy
from hydrax.risk import ExpectedCost, AverageCost, WorstCase, BestCase, ExponentialWeightedAverage, InverseConditionalValueAtRisk, InverseValueAtRisk

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""    
    
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
    actuator_jids = model.jnt_qposadr[actuator_joint_idxs]
    dof_adr  = model.jnt_dofadr[actuator_joint_idxs]    

    # get current joint positions
    qpos = data.qpos.copy()
    qnow = qpos[jnp.array(actuator_jids)]
    qhome = np.array([ 0.51199203,  0.1014329,  -0.36340348, -2.9813132,   0.50339095,  3.06692214, -1.92271156])

    # Build jacobian
    J = np.vstack((jacp, jacr))[:,np.array(dof_adr)]  # (6, n)
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
    
    # print("End effector translation z error:", 0.035 - ee_pos[2])
    temp = np.concatenate([world_site_vel_desired, np.array([0.035-ee_pos[2]])])
    twist_err = np.concatenate([temp, goal_vec])                 # [ex, ey, ez, ewx, ewy, ewz]
    dq = J_pinv @ twist_err

    if with_null_space:
        N = np.eye(J.shape[1]) - J_pinv @ J
        kp_ori = 10.0
        dq += N @ (kp_ori * (qhome - qnow))

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
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.4],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    delay_ctrl_start: int = 0,
    max_step: float = 1e4,
    log_file: str = None,
    record_video: bool = False,
    show_ui: bool = True,
    seed: int = 0,
    dr_strategy: AdaptiveDomainRandomizationStrategy = None,
    trace_idxs=None,
    show_poses: bool = True,
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

    # Warm-up the controller
    # controller.compile_optimize()
    print("Jitting the controller...")
    st = time.time()
    jit_optimize = jit_optimize.lower(mjx_data, policy_params).compile()
    print(f"Time to jit: {time.time() - st:.3f} seconds") 
    
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    print("Sites in rollouts:", rollouts.trace_sites.shape)
    
    if trace_idxs is None:
        num_traces = min(rollouts.controls.shape[1], max_traces)
    else:
        num_traces = len(trace_idxs)

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
        
    # !LIVE PLOT SETUP ------------------------------------------------#
    
    site_id1 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "T_1")
    site_id2 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "T_2")
    site_id3 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    
    # true poses
    old_observation1 = np.concatenate((np.array(mj_data.site_xpos[site_id1]), np.array(mat2quat(mj_data.site_xmat[site_id1])))) 
    old_observation2 = np.concatenate((np.array(mj_data.site_xpos[site_id2]), np.array(mat2quat(mj_data.site_xmat[site_id2]))))
    old_observation3 = np.concatenate((np.array(mj_data.site_xpos[site_id3]), np.array(mat2quat(mj_data.site_xmat[site_id3]))))

    # new_randomizations = dr_strategy.get_current_randomizations()
    sites_of_interest = policy_params.predicted_state
    
    if show_poses:
        from vis_utils import plot_poses_2d
        plt.ion()
        fig, ax_poses = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
        plot_poses_2d(sites_of_interest[...,0,:], ax_poses, ref_pose=old_observation1)
        plt.show(block=False)
    #!---------------------------------------------------------------------------------------#

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
                num_trace_sites * num_traces * controller.num_randomizations * controller.task.planning_horizon
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
            # policy_params, rollouts = controller.opt_step(mjx_data, policy_params)
            plan_time = time.time() - plan_start

            if hasattr(controller, 'beta'):
                controller.beta = float(policy_params.beta) # TODO
                
            
            # ! get error signal
            sites_of_interest = policy_params.predicted_state#[..., 1]  # ignore end effector site
            # true poses
            new_observation1 = jnp.concatenate((jnp.array(mj_data.site_xpos[site_id1]), jnp.array(mat2quat(mj_data.site_xmat[site_id1]))), axis=-1) 
            new_observation2 = jnp.concatenate((jnp.array(mj_data.site_xpos[site_id2]), jnp.array(mat2quat(mj_data.site_xmat[site_id2]))), axis=-1)
            new_observation3 = jnp.concatenate((jnp.array(mj_data.site_xpos[site_id3]), jnp.array(mat2quat(mj_data.site_xmat[site_id3]))), axis=-1)
            
            # check if any change in observation
            if np.allclose(np.array(new_observation1), old_observation1, atol=1e-2) and np.allclose(np.array(new_observation2), old_observation2, atol=1e-2):
                # no change, skip update
                print("No change in T observation, skipping DR update.")
                
            else:
                
                old_observation1 = new_observation1
                old_observation2 = new_observation2

                distance_1 = jax.vmap(se3_left_invariant_metric, in_axes=(0, None))(sites_of_interest[...,0,:], new_observation1)
                distance_2 = jax.vmap(se3_left_invariant_metric, in_axes=(0, None))(sites_of_interest[...,1,:], new_observation2)
                # ee
                distance_3 = jax.vmap(se3_left_invariant_metric, in_axes=(0, None))(sites_of_interest[...,2,:], new_observation3)
                distances = distance_1 + distance_2

                #! post process distances 
                # normalize distances to [0, 1]
                distances = distances - jnp.min(distances)

                # if jnp.max(distances) > 1e-6:
                #     distances = distances / jnp.max(distances)
                
                # clip distances to [0, 1]
                distances = jnp.clip(distances, 0.0, 1.0)
                # set NaN to 1
                max_distance = jnp.max(distances)
                # if max is NaN, set to 1.0
                if not jnp.isfinite(max_distance):
                    max_distance = 1.0
                distances = jnp.nan_to_num(distances, nan=max_distance)
                
                distances = np.array(distances)
                distances = distances / (np.sum(distances) + 1e-12)
                print("Distances:", distances)    

                # scale to [0, 1] for alphas
                alphas = 0.9 * (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-12)    
                alphas = 1 - alphas  # invert, so that smaller distance = higher weight           
                
                # probabilities via softmax
                temperature = np.std(distances) + 1e-12
                probs = np.exp(-distances / temperature)  # temperature scaling
                probs = probs / (np.sum(probs) + 1e-12)

                policy_params = policy_params.replace(domain_weights=jnp.array(probs))
                if show_poses:
                    plot_poses_2d(sites_of_interest[...,0,:], ax=ax_poses, ref_pose=new_observation1, alphas=alphas)
                    fig.canvas.draw_idle()
                    plt.pause(0.01)  # yield to the GUI loop
                # !-------------------------------------------
            
            # Visualize the rollouts
            colors = plt.cm.viridis(np.linspace(0, 1, controller.num_randomizations))
            if trace_idxs is None:
                trace_idxs = list(range(num_traces))
            if show_traces:
                ii = 0
                for k in range(num_trace_sites):
                    for i in trace_idxs:
                        for d, color in enumerate(colors): # num_randomizations
                            for j in range(controller.task.planning_horizon):
                                geom =viewer.user_scn.geoms[ii]
                                mujoco.mjv_connector(
                                    geom,
                                    mujoco.mjtGeom.mjGEOM_LINE,
                                    trace_width,
                                    rollouts.trace_sites[d, i, j, k, :3],        # ! randomizations x rollouts x horizon x sites
                                    rollouts.trace_sites[d, i, j + 1, k, :3],    # !
                                )
                                if k > 0:
                                    geom.rgba = np.array(color.tolist())
                                ii += 1

            
            # Step the simulation
            for i in range(sim_steps_per_replan):
                t = i * mj_model.opt.timestep
                u = controller.get_action(policy_params, t)

                if delay_ctrl_start > 0:
                    delay_ctrl_start -= 1
                    # print(f"Delaying controller start for {delay_ctrl_start} steps")
                else:
                    # remap controls if a control mapper is provided
                    if controller.control_mapper is not None:
                        # print(f"Original control action: {u}")
                        u = differential_IK(
                            mj_model,
                            mj_data,
                            controller.task.ee_body_id,
                            u,  # Exclude base DOF
                        )
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
            if hasattr(controller, 'beta'):
                print(
                    f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s, sim time: {mj_data.time:.2f}s, beta: {controller.beta:.3f}", 
                    end="\r",
                )
            else:
                print(
                    f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s, sim time: {mj_data.time:.2f}s", 
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
                "state_cost": float(rollouts.costs[0, 0]),
                "success": task_success,
                "domain_weights": np.array(policy_params.domain_weights).tolist() if hasattr(controller, 'domain_weights') else None,
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
        log_dir = Path(log_file)
        log_dir.mkdir(parents=True, exist_ok=True)   # create directory if missing
        
        with open(log_file, "wb") as f:
            pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"State bins saved to {log_file}")