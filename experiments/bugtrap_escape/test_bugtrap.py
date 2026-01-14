import mujoco
import mujoco.viewer
from mujoco import mj_id2name, mj_name2id
import os
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


# xml_path = "/home/magnus/OneDrive/Dokumente/Uni/Master/Thesis/mjx/models/scene_real.xml"
xml_path = "/home/magnus/GitHub/mtp-thesis/hydrax/models/bugtrap/scene.xml"
xml_dir = os.path.dirname(xml_path)

# Change working directory temporarily
os.chdir(xml_dir)
model = mujoco.MjModel.from_xml_path(os.path.basename(xml_path))
data = mujoco.MjData(model)

# Disable gravity
# model.opt.gravity[:] = 0.0

# Some model info
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
    step = 0
    while v.is_running():
        # Advance simulation (computes contacts etc.)
        mujoco.mj_step(model, data)
        # Check number of contacts
        
        if step % 10 == 0:
            print("ncon:", data.ncon)

            cfrc_ext = data.cfrc_ext.reshape(model.nbody, 6)
            for body_id in range(model.nbody):
                wrench = cfrc_ext[body_id]
                body_name = model.body(body_id).name
                print(f"{body_id:2d} {body_name:15s}: {wrench}")
            for i in range(data.ncon):
                con = data.contact[i]
                print("contact", i,
                    "geom1:", model.geom(con.geom1).name,
                    "geom2:", model.geom(con.geom2).name)
        # data.ctrl[0] = 0.01
        v.sync()
        step += 1