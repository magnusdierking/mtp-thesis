import os
import sys
import time

import keyboard
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mj_id2name, mj_name2id
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from hydrax.utils.utils import (
    euler_to_quaternion,
    mat2quat,
    mujoco_to_scipy_quat,
    quat_conj,
    quat_error_body,
    quat_mul,
    quat_normalize,
    quat_to_rotvec,
    se3_left_invariant_metric,
)


def differential_IK(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: str = "ee_frame",
    world_site_vel_desired: np.ndarray = np.zeros(2),
    with_null_space: bool = True,
) -> np.ndarray:
    """Differential IK for all dofs in the model."""
    # Geometric Jacobians at body_id
    jacp = np.zeros((3, model.nv), dtype=np.float64)  # translational
    jacr = np.zeros((3, model.nv), dtype=np.float64)  # rotational

    bodyid = model.body("ee_frame").id
    mujoco.mj_jacBody(model, data, jacp, jacr, bodyid)

    actuator_joint_names = [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ]
    actuator_joint_idxs = [model.joint(name).id for name in actuator_joint_names]
    actuator_jids = model.jnt_qposadr[actuator_joint_idxs]
    dof_adr = model.jnt_dofadr[actuator_joint_idxs]

    # get current joint positions
    qpos = data.qpos.copy()
    qnow = qpos[np.array(actuator_jids)]
    qhome = np.array(
        [
            0.51199203,
            0.1014329,
            -0.36340348,
            -2.9813132,
            0.50339095,
            3.06692214,
            -1.92271156,
        ]
    )

    # Build jacobian
    J = np.vstack((jacp, jacr))[:, np.array(dof_adr)]  # (6, n)
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
    goal_quat = np.array(
        [0.0, 0.7071, 0.7071, 0.0]
    )  # ([0.0, 0.0, 0.7071, 0.7071])  # Assuming goal orientation is aligned with x-axis
    goal_quat = np.array(goal_quat)
    goal_vec = quat_error_body(goal_quat, ee_quat)  # (3,)

    temp = np.concatenate([world_site_vel_desired, np.array([0.035 - ee_pos[2]])])
    twist_err = np.concatenate([temp, goal_vec])  # [ex, ey, ez, ewx, ewy, ewz]
    dq = J_pinv @ twist_err

    if with_null_space:
        N = np.eye(J.shape[1]) - J_pinv @ J
        kp_ori = 10.0
        dq += N @ (kp_ori * (qhome - qnow))

    return dq

# ------------------------------ #
det_init = {
        "block_pos_x": 0.6,
        "block_pos_y": 0.05,
        "block_angle": 5*np.pi/4,
        "ee_goal_pos": [0.45, 0.1, 0.035]
    }




xml_path = "./../../hydrax/models/fr3_pushT_vel/scene_mjx_free.xml"
xml_dir = os.path.dirname(xml_path)


# Change working directory temporarily
os.chdir(xml_dir)
model = mujoco.MjModel.from_xml_path(os.path.basename(xml_path))
data = mujoco.MjData(model)

# model.opt.gravity[:] = 0.0
print(len(data.qpos))
print(len(data.qvel))
print("===========")

for i in range(model.njnt):
    name = model.joint(i).name
    print(f"Joint {i}: {name}")
    # joint index
    print(f"  Joint index: {model.joint(i).id}")
    # qpos address
    print(f"  qpos address: {model.jnt_qposadr[i]}")
    # qvel address
    print(f"  qvel address: {model.jnt_dofadr[i]}")

for i in range(model.nsite):
    name = model.site(i).name
    print(f"Site {i}: {name}")

for bid in range(model.nbody):
    mocap_id = model.body_mocapid[bid]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    mass = model.body_mass[bid]
    print(f"body_id={bid}, name={name}, mass={mass}, mocap_id={mocap_id}")


# # Initial guess
q = np.array([0.0, -np.pi / 4, 0.0, -9 * np.pi / 10, 0.0, 3 * np.pi / 4, np.pi / 4])
# q = np.zeros(7)
actuator_joint_names = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]
actuator_joint_idxs = [model.joint(name).id for name in actuator_joint_names]
actuator_jids = model.jnt_qposadr[actuator_joint_idxs]
dof_adr = model.jnt_dofadr[actuator_joint_idxs]

ee_body_id = model.body("ee_frame").id
goal_quat_ee = np.array([0.0, 0.7071, 0.7071, 0.0])  # [w, x, y, z]
goal_pos_ee = np.array([det_init["ee_goal_pos"][0],
                        det_init["ee_goal_pos"][1],
                        0.045])  # np.array([0.3, 0.0, 0.05])
joint_limits = model.jnt_range[actuator_joint_idxs]

# IK loop parameters
max_iters = 2_000
tolerance = 1e-3
damping = 100e-3
step_size = 1.0

ik_start_time = time.time()
for i in range(max_iters):
    # Set current joint sdtate
    data.qpos[actuator_jids] = q
    mujoco.mj_forward(model, data)

    # Current EE pose
    current_pos = data.xpos[ee_body_id]
    current_quat = data.xquat[ee_body_id]

    # Position error
    pos_err = goal_pos_ee - current_pos  # shape (3,)

    # Orientation error (quaternion distance -> angular velocity vector)
    r_current = R.from_quat(mujoco_to_scipy_quat(current_quat))
    r_desired = R.from_quat(mujoco_to_scipy_quat(goal_quat_ee))

    # Rotation needed to go from current to desired
    r_error = r_desired * r_current.inv()

    # Convert to rotation vector (axis-angle * angle)
    orn_err = r_error.as_rotvec()  # shape (3,)

    # Combined 6D task error
    err = np.concatenate([pos_err, orn_err])  # shape (6,)

    if np.linalg.norm(err) < tolerance:
        print(f"Converged in {i} iterations. Took {time.time() - ik_start_time:.4f} s")
        break

    # Compute Jacobian of the EE
    J_pos = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, J_pos, J_rot, ee_body_id)

    # Slice columns corresponding to actuated joints
    J = np.vstack([J_pos[:, dof_adr], J_rot[:, dof_adr]])  # shape (6, n_joints)

    # Solve damped least squares: dq = (JᵀJ + λ²I)⁻¹ Jᵀ e
    JTJ = J.T @ J
    H = JTJ + damping * np.eye(len(dof_adr))
    g = J.T @ err
    dq = np.linalg.solve(H, g)

    # Update joint configuration
    q += step_size * dq

    # Clamp to joint limits
    q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])

else:
    print(f"IK did not converge. {err}, {np.linalg.norm(err)}")


data.qpos[actuator_jids] = q  # Set the robot's joint positions


## T POSITIONING THE BLOCK
data.qpos[0] = det_init["block_pos_x"]
data.qpos[1] = det_init["block_pos_y"]
data.qpos[2] = 0.08  # block z position
angle = det_init["block_angle"]

quat = euler_to_quaternion(0, 0, angle)  # x,y,z,w
print("Block quaternion (x,y,z,w):", quat[0], quat[1], quat[2], quat[3])
data.qpos[3:7] = np.array([quat[3], quat[0], quat[1], quat[2]])  # x, y, z, w


bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
print("mass:", model.body_mass[bid])
print("diaginertia:", model.body_inertia[bid])
print("inertial quat:", model.body_ipos[bid], model.body_iquat[bid])  # COM + quat


mujoco.mj_step(model, data)

step = 0
scale = 0.8
scaled = False

site_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "T_1")


def quat_to_yaw(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny, cosy)


# Oreintation sensor id
block_orientation_sensor = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation_site"
)
sensor_adr_orientation = model.sensor_adr[block_orientation_sensor]

# Position sensor id
block_position_sensor = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SENSOR, "position_site"
)
sensor_adr_position = model.sensor_adr[block_position_sensor]


speed = 0.2
with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        vel_x = (
            speed
            if keyboard.is_pressed("right")
            else -speed
            if keyboard.is_pressed("left")
            else 0.0
        )
        vel_y = (
            speed
            if keyboard.is_pressed("up")
            else -speed
            if keyboard.is_pressed("down")
            else 0.0
        )
        mujoco.mj_step(model, data)

        dq = differential_IK(
            model,
            data,
            body_id="ee_frame",
            world_site_vel_desired=np.array([vel_x, vel_y]),
            with_null_space=True,
        )
        data.ctrl[:] = dq
        step += 1

        # sys.stdout.write(f"\rQuaternion: {block_quat}, Position: {block_pos} ")
        # sys.stdout.write(f"\rncon: {data.ncon} | nj: {data.nJ} | time: {data.time:.4f}s | ee vel: {vel} ")
        # sys.stdout.flush()

        v.sync()
