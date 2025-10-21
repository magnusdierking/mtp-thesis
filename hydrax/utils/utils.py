import numpy as np

def mujoco_to_scipy_quat(q):
    return np.array([q[1], q[2], q[3], q[0]])
