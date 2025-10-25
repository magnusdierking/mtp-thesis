import time
from typing import Sequence
import csv

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

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""    
    
def differential_IK(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: str = "ee_frame",
    world_site_vel_desired: np.ndarray = np.zeros(2),
) -> np.ndarray:
    """
    Differential IK for all dofs in the model.
    """
    # Geometric Jacobians at body_id
    jacp = np.zeros((3, model.nv), dtype=np.float64)  # translational
    jacr = np.zeros((3, model.nv), dtype=np.float64)  # rotational

    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    # Build jacobian
    J = np.vstack((jacp, jacr))  # (6, n)

    # Compute dq with damped pseudo-inverse
    J_pseudo_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
    twist = np.concatenate([world_site_vel_desired, np.zeros(4)])
    dq = J_pseudo_inv @ twist

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
        
    # !LIVE PLOT SETUP ------------------------------------------------#
    from scipy.stats import gaussian_kde
    from collections import deque
    plt.ion()
    fig, (ax_bar, ax_kde) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # --- BAR CHART (left) ---
    values = np.arange(controller.num_randomizations)
    probs = np.ones(controller.num_randomizations) / controller.num_randomizations

    bars = ax_bar.bar(values, probs, width=0.8, align="center", edgecolor="k")
    ax_bar.set_xticks(values)
    ax_bar.set_xticklabels([f"{v:.2f}" for v in probs])
    ax_bar.set_xlabel("Outcome")
    ax_bar.set_ylabel("Probability")
    ax_bar.set_title("Discrete Distribution")

    # --- KDE PLOT (right) ---
    # a rolling buffer of samples to build the KDE from (set maxlen=None to keep all)
    kde_samples = deque(maxlen=2000)  # adjust if you want a rolling window
    kde_samples.extend(np.asarray(controller.model.body_mass[:,controller.task.T_bid].tolist(), dtype=float).ravel()) 

    # initialize an empty line for the KDE
    samples_array = np.fromiter(kde_samples, dtype=float)

    # Build KDE (adjust bw_method to taste: 'scott', 'silverman', or a float)
    kde = gaussian_kde(samples_array, bw_method='scott')

    # Grid for evaluation — pad a bit beyond min/max to avoid clipping
    s_min, s_max = float(samples_array.min()), float(samples_array.max())
    pad = 0.05 * (s_max - s_min if s_max > s_min else max(s_max, 1.0))
    x_kde = np.linspace(s_min - pad, s_max + pad, 512)
    y_kde = kde(x_kde)
    kde_line, = ax_kde.plot([], [], lw=2)

    # Update the line
    kde_line.set_data(x_kde, y_kde)
    ax_kde.set_xlim(x_kde[0], x_kde[-1])
    ax_kde.set_ylim(0, max(y_kde) * 1.05 if np.isfinite(y_kde).any() else 1.0)
    # kde_line, = ax_kde.plot([], [], lw=2)
    ax_kde.set_xlabel("Sample value")
    ax_kde.set_ylabel("Density")
    ax_kde.set_title("KDE (updates with new samples)")

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
                
            # !update live plot
            sites_of_interest = policy_params.predicted_state[..., 1]  # ignore end effector site
            site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "T_1")

            distances = np.linalg.norm(
                sites_of_interest - np.array(mj_data.site_xpos[site_id]), axis=-1
            )

            # probs based on distances
            z = distances / 0.008
            exps = np.exp(z - np.max(z))  # for numerical stability
            probs = exps / np.sum(exps)

            # --- update bar heights (left subplot) ---
            for rect, h in zip(bars, probs):
                rect.set_height(h)

            ax_bar.set_xticklabels(
                [f"{v:.2f}" for v in controller.model.body_mass[:, controller.task.T_bid]]
            )

            ax_bar.relim()
            ax_bar.autoscale_view(scaley=True)


            # Only build a KDE once we have at least 2 samples
            if len(kde_samples) >= 2:
                samples_array = np.fromiter(kde_samples, dtype=float)

                # Build KDE (adjust bw_method to taste: 'scott', 'silverman', or a float)
                kde = gaussian_kde(samples_array, bw_method='scott')

                # Grid for evaluation — pad a bit beyond min/max to avoid clipping
                s_min, s_max = float(samples_array.min()), float(samples_array.max())
                pad = 0.05 * (s_max - s_min if s_max > s_min else max(s_max, 1.0))
                x_kde = np.linspace(s_min - pad, s_max + pad, 512)
                y_kde = kde(x_kde)

                # Update the line
                kde_line.set_data(x_kde, y_kde)
                ax_kde.set_xlim(x_kde[0], x_kde[-1])
                ax_kde.set_ylim(0, max(y_kde) * 1.05 if np.isfinite(y_kde).any() else 1.0)
            else:
                # Not enough samples yet — clear the line
                kde_line.set_data([], [])
                ax_kde.set_ylim(0, 1)

            # ----------------------------------------------------------------------- #

            fig.canvas.draw_idle()
            plt.pause(0.01)  # yield to the GUI loop
            
            # update domain randomizations
            updated = controller.update_domain_randomization_model(
                jax.random.PRNGKey(step), jnp.array(probs)
            )
            if updated:
                kde_samples.extend(np.asarray(controller.model.body_mass[:,controller.task.T_bid].tolist(), dtype=float).ravel())  
        
            # !-------------------------------------------
            
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

            # for k in range(controller.nbr_actions):
            
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
                        u = differential_IK(
                            mj_model,
                            mj_data,
                            controller.task.ee_body_id,
                            u,  # Exclude base DOF
                        )
                    # print(f"Remapped control action: {u}")
                    
                    if controller.gravity_compensator:
                        # Gravity compensation for the robot only
                        tau_g = gravity_comp_torque(mj_model, mj_data)
                        # Clear and apply external torques (generalized forces)
                        mj_data.qfrc_applied[:] = 0.0        # clears all user generalized forces
                        mj_data.xfrc_applied[:] = 0.0        # clears any body-space external wrenches
                        mj_data.qfrc_applied[controller.task.actuator_joint_idxs] = tau_g[controller.task.actuator_joint_idxs]
                        # Apply the control to the simulation
                    mj_data.ctrl[:] = np.array(u[np.array(controller.task.actuator_joint_idxs)])
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