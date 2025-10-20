from pyexpat import model
import time
import numpy as np
from hydrax.simulation.asynchronous import run_interactive
from hydrax.tasks.pusht_franka import PushTFranka
import mujoco
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.linalg import null_space
from scipy.spatial.transform import Rotation as R


def differential_IK(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: str = "ee_frame",
    world_site_vel_desired: np.ndarray = np.zeros(6),
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
    dq = J_pseudo_inv @ world_site_vel_desired

    return dq


def gravity_comp_torque(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    # Backup velocities
    qvel_backup = data.qvel.copy()

    # Zero velocities so qfrc_bias reduces to pure gravity
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    tau_g = data.qfrc_bias.copy()  # generalized coordinates (size nv)

    # Restore velocities and recompute
    data.qvel[:] = qvel_backup
    mujoco.mj_forward(model, data)

    return tau_g


if __name__ == "__main__":
    
    seed = 0

    task = PushTFranka(actuation_type='velocity')

    mj_model, mj_data = task.reset(seed=seed)
    
    disable_gravity = True
    if disable_gravity:
        mj_model.opt.gravity[:] = [0., 0., 0.]
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
        mujoco.mj_forward(mj_model, mj_data)
    
    ee_initial_quat = mj_data.xquat[task.ee_body_id]
    ee_initial_euler = R.from_quat(ee_initial_quat, scalar_first=True).as_euler('xyz', degrees=True)

    planning_frequency = 250.0  # Hz

    # How many sim steps to run before replanning
    replan_period = 1.0 / planning_frequency
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Sim Step/Replan {sim_steps_per_replan} steps, "
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0/mj_model.opt.timestep} Hz"
    )

    motion_type = 'x'

    with mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=True, show_right_ui=True) as viewer:

        # --- init ---
        step = 0
        simulation_start_time = time.time()
        motion_type_time = time.time()
        while viewer.is_running():
            step += 1
            start_time = time.time()

            # Step the simulation
            for i in range(sim_steps_per_replan):
                t = i * mj_model.opt.timestep

                # Tangential EE velocity
                radius = 0.1
                omega = 0.3 * 2 * np.pi  # rad/s
                if motion_type == 'x':
                    dx = radius * omega * np.sin(omega * (time.time() - simulation_start_time))  # vx
                    dy = 0
                else:
                    dx = 0
                    dy = -radius * omega * np.cos(omega * (time.time() - simulation_start_time))  # vy
                a = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0])

                dq = differential_IK(
                    mj_model,
                    mj_data,
                    task.ee_body_id,
                    a,
                )
                
                # Gravity compensation for the robot only
                if not disable_gravity:
                    tau_g = gravity_comp_torque(mj_model, mj_data)
                    # Clear and apply external torques (generalized forces)
                    mj_data.qfrc_applied[:] = 0.0        # clears all user generalized forces
                    mj_data.xfrc_applied[:] = 0.0        # clears any body-space external wrenches
                    mj_data.qfrc_applied[task.actuator_joint_idxs] = tau_g[task.actuator_joint_idxs]

                # Apply the control to the simulation
                mj_data.ctrl[:] = np.array(dq[task.actuator_joint_idxs])
                
                # For a safe check set the velocities directly (bypassing actuator model)
                # and zero out controls and forces
                # mj_data.qvel[task.actuator_joint_idxs] = dq[task.actuator_joint_idxs]
                # mj_data.qacc[:] = 0.0
                # mj_data.ctrl[:] = 0.0
                # mj_data.qfrc_applied[:] = 0.0
                # mj_data.xfrc_applied[:] = 0.0
                
                
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                
            # Try to run in roughly realtime
            elapsed_time = time.time() - start_time
            if elapsed_time < step_dt:
                time.sleep(step_dt - elapsed_time)
                
            # Check the EE orientation did not drift
            ee_current_quat = mj_data.xquat[task.ee_body_id]
            ee_current_euler = R.from_quat(ee_current_quat, scalar_first=True).as_euler('xyz', degrees=True)
            ee_drift_euler = ee_current_euler - ee_initial_euler
            msg = f"EE euler drift: {ee_drift_euler}"

            # Print some information
            real_time_rate = step_dt / (time.time() - start_time)
            msg += f" | Realtime rate: {real_time_rate:.2f}"

            print(msg, end="\r")


