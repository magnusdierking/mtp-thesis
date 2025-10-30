import numpy as np
import jax
import jax.numpy as jnp

def mujoco_to_scipy_quat(q):
    return np.array([q[1], q[2], q[3], q[0]])


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