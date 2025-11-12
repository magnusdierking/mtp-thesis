import mujoco
import mujoco.viewer
from mujoco import mj_id2name, mj_name2id
import os
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import time
from hydrax.utils.utils import mujoco_to_scipy_quat


# xml_path = "./../hydrax/models/g1/scene.xml"
xml_path = "./../hydrax/models/fr3_pushT_vel/scene_mjx_free.xml"
xml_dir = os.path.dirname(xml_path)

# Change working directory temporarily
os.chdir(xml_dir)
model = mujoco.MjModel.from_xml_path(os.path.basename(xml_path))
# model.opt.gravity[:] = 0 
data = mujoco.MjData(model)

# model.opt.gravity[:] = 0.0
print(len(data.qpos))

for i in range(model.njnt):
    name = model.joint(i).name
    print(f"Joint {i}: {name}")
for i in range(model.nu):
    name = mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"Control input {i}: {name}")
for i in range(model.nsite):
    name = model.site(i).name
    print(f"Site {i}: {name}")
    
    
    

        # Assuming the block's pose is at the beginning of qpos


# # Initial guess
q = np.array([0.0, -np.pi/4, 0.0, -9*np.pi/10, 0.0, 3*np.pi/4, np.pi/4])
# q = np.zeros(7)
actuator_joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
actuator_joint_idxs = [model.joint(name).id for name in actuator_joint_names]
actuator_jids = model.jnt_qposadr[actuator_joint_idxs]
dof_adr  = model.jnt_dofadr[actuator_joint_idxs] 
print("Actuator joint ids:", actuator_jids)

ee_body_id = model.body("ee_frame").id
goal_quat_ee = np.array([0.0, 0.7071, 0.7071, 0.0])  # [w, x, y, z]
goal_pos_ee = np.array([0.6, 0.0, 0.035]) #np.array([0.3, 0.0, 0.05])
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
    

    
with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        # data.ctrl[0] = 0.01 #q
        mujoco.mj_step(model, data)
        print("EE pos:", data.xpos[ee_body_id])
        print("EE quat:", data.xquat[ee_body_id])
        v.sync()
