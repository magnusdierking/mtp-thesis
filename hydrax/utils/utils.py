import math

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx


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
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_error_body(qd, q):
    """Right-invariant error q_e = qd * q^{-1} (error in current/body frame)."""
    qd = quat_normalize(qd)
    q = quat_normalize(q)
    qe = quat_mul(qd, quat_conj(q))
    # Enforce shortest rotation (w >= 0)
    return quat_to_rotvec(jnp.where(qe[0] < 0.0, -qe, qe))


def quat_to_rotvec(qe, eps=1e-8):
    """Quaternion (unit) -> rotation vector (axis * angle)."""
    w, x, y, z = qe
    w = jnp.clip(w, -1.0, 1.0)
    angle = 2.0 * jnp.arccos(w)
    s = jnp.sqrt(1.0 - w * w)
    axis = jnp.where(s < eps, jnp.array([1.0, 0.0, 0.0]), jnp.array([x, y, z]) / s)
    return angle * axis


def mat2quat(mat):
    mat = mat.reshape(3, 3)
    # transform 3x3 rotation matrix to quaternion
    w = jnp.sqrt(1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2]) / 2.0
    x = (mat[2, 1] - mat[1, 2]) / (4.0 * w)
    y = (mat[0, 2] - mat[2, 0]) / (4.0 * w)
    z = (mat[1, 0] - mat[0, 1]) / (4.0 * w)
    return jnp.array([w, x, y, z])


def quat2mat(q):
    w, x, y, z = q
    return jnp.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


# SE3 left invariant metric
def se3_left_invariant_metric(p1, p2, rot_weight=1.0, trans_weight=10.0):
    """Compute left-invariant metric between two SE(3) poses.

    Args:
        p1: First pose, shape (..., 7) (x, y, z, qw, qx, qy, qz).
        p2: Second pose, shape (..., 7) (x, y, z, qw, qx, qy, qz).
        rot_weight: Weight for rotational component.
        trans_weight: Weight for translational component.
    Returns:
        The left-invariant distance between p1 and p2, shape (...,).
    """
    # jax.debug.print("se3_left_invariant_metric called with p1 shape: {}, p2 shape: {}", p1.shape, p2.shape)
    pos1, quat1 = p1[:3], p1[3:]
    pos2, quat2 = p2[:3], p2[3:]

    return rot_weight * jnp.linalg.norm(
        mjx._src.math.quat_sub(quat1, quat2)
    ) + trans_weight * jnp.linalg.norm(pos2 - pos1)
