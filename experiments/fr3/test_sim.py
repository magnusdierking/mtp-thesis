import mujoco
import mujoco.viewer
from mujoco import mj_id2name, mj_name2id
import os
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from hydrax.utils.utils import mujoco_to_scipy_quat


# Load the MuJoCo model
xml_path = "./../../hydrax/models/fr3_pushT_vel/scene_mjx.xml"
# xml_path = "./../../hydrax/models/pusht_franka_velocity/scene_mjx.xml"
# xml_path = "/home/magnus/GitHub/mtp/hydrax/hydrax/models/pusht_franka/scene.xml"
xml_dir = os.path.dirname(xml_path)

# Change working directory temporarily
os.chdir(xml_dir)
model = mujoco.MjModel.from_xml_path(os.path.basename(xml_path))
model.opt.gravity[:] = 0 
data = mujoco.MjData(model)

# model.opt.gravity[:] = 0.0

for i in range(model.njnt):
    name = model.joint(i).name
    print(f"Joint {i}: {name}")
for i in range(model.nu):
    name = mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"Control input {i}: {name}")
for i in range(model.nsite):
    name = model.site(i).name
    print(f"Site {i}: {name}")
    
print(data.ctrl)
                         
                         # initial position of robot
# Desired EE pose in world frame
des_pos = np.array([0.3, 0.0, 0.035])
des_quat = np.array([0.0, 0.7071, 0.7071, 0.0])  # [w, x, y, z]

# Body ID for end-effector
body_id = model.body("ee_frame").id

# Joint index range (skip floating base joints if any)
j_start = 3  # Adjust based on your model (e.g., 3 if floating base)
n_joints = 7

# Joint limits
joint_limits = np.array([model.jnt_range[i] for i in range(j_start, model.njnt)])

# # Initial guess
q = np.array([0.0, -np.pi/4, 0.0, -9*np.pi/10, 0.0, 3*np.pi/4, np.pi/4])

# IK loop parameters
max_iters = 300
tolerance = 1e-3
damping = 50e-3

for i in range(max_iters):
    # Set current joint state
    data.qpos[j_start:j_start+n_joints] = q
    mujoco.mj_forward(model, data)

    # Current EE pose
    current_pos = data.xpos[body_id]
    current_quat = data.xquat[body_id]

    # Position error
    pos_err = des_pos - current_pos  # shape (3,)

    # Orientation error (quaternion distance -> angular velocity vector)
    r_current = R.from_quat(mujoco_to_scipy_quat(current_quat))
    r_desired = R.from_quat(mujoco_to_scipy_quat(des_quat))

    # Rotation needed to go from current to desired
    r_error = r_desired * r_current.inv()

    # Convert to rotation vector (axis-angle * angle)
    orn_err = r_error.as_rotvec()  # shape (3,)

    # Combined 6D task error
    err = np.concatenate([pos_err, orn_err])  # shape (6,)

    if np.linalg.norm(err) < tolerance:
        print(f"Converged in {i} iterations.")
        break

    # Compute Jacobian of the EE
    J_pos = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, J_pos, J_rot, body_id)

    # Slice columns corresponding to actuated joints
    J = np.vstack([J_pos[:, j_start:j_start+n_joints], J_rot[:, j_start:j_start+n_joints]])  # shape (6, n_joints)

    # Solve damped least squares: dq = (JᵀJ + λ²I)⁻¹ Jᵀ e
    JTJ = J.T @ J
    H = JTJ + damping * np.eye(n_joints)
    g = J.T @ err
    dq = np.linalg.solve(H, g)

    # Update joint configuration
    q += dq

    # Clamp to joint limits
    for j in range(n_joints):
        low, high = joint_limits[j]
        q[j] = np.clip(q[j], low, high)

else:
    print("IK did not converge.")


# initial control
data.qpos[0] = 0.1
data.qpos[1] = -0.1
data.qpos[j_start:j_start+n_joints] = q
data.qvel[:] = 0.0
data.ctrl[:] = 0 #q
mujoco.mj_forward(model, data) 



# print everything
mujoco.mj_printModel(model, '/tmp/model.txt')
    
    
with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():

        mujoco.mj_step(model, data)
        v.sync()
