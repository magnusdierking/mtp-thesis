import mujoco
import mujoco.viewer
from mujoco import mj_id2name, mj_name2id
import os
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# xml_path = "/home/magnus/OneDrive/Dokumente/Uni/Master/Thesis/mjx/models/scene_real.xml"
xml_path = "/home/magnus/GitHub/mtp/hydrax/hydrax/models/bugtrap/scene.xml"
xml_dir = os.path.dirname(xml_path)

# Change working directory temporarily
os.chdir(xml_dir)
model = mujoco.MjModel.from_xml_path(os.path.basename(xml_path))
data = mujoco.MjData(model)

model.opt.gravity[:] = 0.0

for i in range(model.njnt):
    name = model.joint(i).name
    print(f"Joint {i}: {name}")
for i in range(model.nu):
    name = mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"Control input {i}: {name}")
for i in range(model.nsite):
    name = model.site(i).name
    print(f"Site {i}: {name}")
    

                    
    
with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():

        mujoco.mj_step(model, data)
        v.sync()
