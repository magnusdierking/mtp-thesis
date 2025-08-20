import mujoco
import mujoco.viewer
from mujoco import mj_id2name, mj_name2id
import os
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R



xml_path = "/home/magnus/GitHub/mtp/hydrax/hydrax/models/unitree_go2/scene.xml"
xml_dir = os.path.dirname(xml_path)

# Change working directory temporarily
os.chdir(xml_dir)
model = mujoco.MjModel.from_xml_path(os.path.basename(xml_path))
data = mujoco.MjData(model)

# print initial state
print("Initial state:")
for i in range(model.nu):
    print(f"Joint {i}: {data.qpos[i]}, Velocity: {data.qvel[i]}")
    name = mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    
# print control inputs
for i in range(model.nu):
    name = mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"Control input {i}: {name}")


    
with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        # print joints
        mujoco.mj_step(model, data)
        v.sync()
