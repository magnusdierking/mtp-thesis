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



def quat_normalize(q):
    return q / jnp.linalg.norm(q)

def quat_conj(q):  # [w, x, y, z]
    w, x, y, z = q
    return jnp.array([w, -x, -y, -z])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_error_body(qd, q):
    """Right-invariant error q_e = qd * q^{-1} (error in current/body frame)."""
    qd = quat_normalize(qd)
    q  = quat_normalize(q)
    qe = quat_mul(qd, quat_conj(q))
    # Enforce shortest rotation (w >= 0)
    return quat_to_rotvec(jnp.where(qe[0] < 0.0, -qe, qe))

def quat_to_rotvec(qe, eps=1e-8):
    """Quaternion (unit) -> rotation vector (axis * angle)."""
    w, x, y, z = qe
    w = jnp.clip(w, -1.0, 1.0)
    angle = 2.0 * jnp.arccos(w)
    s = jnp.sqrt(1.0 - w*w)
    axis = jnp.where(s < eps, jnp.array([1.0, 0.0, 0.0]), jnp.array([x, y, z]) / s)
    return angle * axis

# def differential_IK(
#     model: mujoco.MjModel,
#     data: mujoco.MjData,
#     body_id: str = "ee_frame",
#     world_site_vel_desired: np.ndarray = np.zeros(6),
# ) -> np.ndarray:
#     """
#     Differential IK for all dofs in the model.
#     """
#     # Geometric Jacobians at body_id
#     jacp = np.zeros((3, model.nv), dtype=np.float64)  # translational
#     jacr = np.zeros((3, model.nv), dtype=np.float64)  # rotational

#     mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

#     # Build jacobian
#     J = np.vstack((jacp, jacr))  # (6, n)

#     # Compute dq with damped pseudo-inverse
#     J_pseudo_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
#     dq = J_pseudo_inv @ world_site_vel_desired

#     return dq

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

    bodyid = model.body("ee_frame").id
    mujoco.mj_jacBody(model, data, jacp, jacr, bodyid)

    actuator_joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
    actuator_joint_idxs = [model.joint(name).id for name in actuator_joint_names]

    # J = J[:, actuator_joint_idxs]        # (6, n_act)
    jacpr = jacp[:, actuator_joint_idxs]
    jacrr = jacr[:, actuator_joint_idxs]

    # Build jacobian
    J = np.vstack((jacp, jacr))  # (6, n)
    print("J shape:", J.shape)
    J_pinv = np.linalg.pinv(J)
    twist = np.concatenate([world_site_vel_desired, np.zeros(4)])
    print("twist shape:", twist.shape)

    N = np.eye(J.shape[1]) - J_pinv @ J
    kp_ori = 10.0

    # J_pseudo_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
    
    # dq = J_pseudo_inv @ twist

    ee_orientation_sensor = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_quat"
        )
    sensor_adr = model.sensor_adr[ee_orientation_sensor]
    ee_quat = data.sensordata[sensor_adr : sensor_adr + 4]
    ee_quat = np.array(ee_quat)
    goal_quat = np.array([0.0, 0.7071, 0.7071, 0.0])  #([0.0, 0.0, 0.7071, 0.7071])  # Assuming goal orientation is aligned with x-axis
    goal_quat = np.array(goal_quat)
    goal_vec = quat_error_body(goal_quat, ee_quat)                                   # (3,)

    temp = np.concatenate([world_site_vel_desired, np.zeros(1)])
    twist_err = np.concatenate([temp, goal_vec])                 # [ex, ey, ez, ewx, ewy, ewz]
    dq = J_pinv @  twist_err #twist + N @ (kp_ori * J.T @ twist_err)
    # #
    # N = jnp.eye(J.shape[1]) - jnp.linalg.pinv(J) @ J
    # kp_ori = 2.0
    # dq = -jnp.linalg.pinv(J) @ (twist_cmd) + N @ (kp_ori * J.T @ twist_err)
    #  twist = twist_cmd + twist_err
    # twist = np.concatenate([world_site_vel_desired, np.zeros(4)])

    # dq = J_pseudo_inv @ twist

    return dq


# def gravity_comp_torque(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
#     # Backup velocities
#     qvel_backup = data.qvel.copy()

#     # Zero velocities so qfrc_bias reduces to pure gravity
#     data.qvel[:] = 0.0
#     mujoco.mj_forward(model, data)
#     tau_g = data.qfrc_bias.copy()  # generalized coordinates (size nv)

#     # Restore velocities and recompute
#     data.qvel[:] = qvel_backup
#     mujoco.mj_forward(model, data)

#     return tau_g

def gravity_comp_torque(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    qvel_bak, qacc_bak = data.qvel.copy(), data.qacc.copy()
    data.qvel[:] = 0.0; data.qacc[:] = 0.0
    tau = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, tau)  # tau = g(q) via inverse dynamics solver
    data.qvel[:] = qvel_bak; data.qacc[:] = qacc_bak
    mujoco.mj_forward(model, data)
    return tau


if __name__ == "__main__":
    
    seed = 154

    task = PushTFranka(actuation_type='velocity')

    mj_model, mj_data = task.reset(seed=seed)
    
    disable_gravity = False
    if disable_gravity:
        mj_model.opt.gravity[:] = [0., 0., 0.]
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
        mujoco.mj_forward(mj_model, mj_data)
    
    ee_initial_quat = mj_data.xquat[task.ee_body_id]
    ee_initial_euler = R.from_quat(ee_initial_quat, scalar_first=True).as_euler('xyz', degrees=True)

    planning_frequency = 100.0  # Hz

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
                radius = 0.3
                omega = 0.05 * 2 * np.pi  # rad/s
                if motion_type == 'x':
                    dx = radius * omega * np.sin(omega * (time.time() - simulation_start_time))  # vx
                    dy = 0
                else:
                    dx = 0
                    dy = -radius * omega * np.cos(omega * (time.time() - simulation_start_time))  # vy
                # a = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0])
                a = np.array([dx, dy])
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
            
            # print simulation time
            sim_time = mj_data.time
            msg += f" | Sim time: {sim_time:.2f}s"

            print(msg, end="\r")


