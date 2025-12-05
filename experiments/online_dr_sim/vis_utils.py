import jax.numpy as jnp
import matplotlib.pyplot as plt

def quat_to_yaw(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return jnp.arctan2(siny, cosy) + jnp.pi/1.0  # Adjusted by +90 degrees to align with plotting convention


def plot_poses_2d(poses, ax, ref_pose=None, alphas=None, arrow_length=0.05):
    """
    Always clears the axis and redraws the poses from scratch.
    Old lines never accumulate.
    """
    ax.cla()

    # Extract components
    x = -jnp.asarray(poses[:, 1])
    y = jnp.asarray(poses[:, 0])
    qw = poses[:, 3]
    qx = poses[:, 4]
    qy = poses[:, 5]
    qz = poses[:, 6]

    yaw = quat_to_yaw(qx, qy, qz, qw)

    if alphas is None:
        alphas = jnp.ones_like(x)

    # Scatter
    scatter = ax.scatter(x, y, s=30)
    scatter.set_facecolors([(0, 0, 1, float(a)) for a in alphas])

    # Draw arrows
    for xi, yi, yiw, a in zip(x, y, yaw, alphas):
        c = float(jnp.cos(yiw))
        s = float(jnp.sin(yiw))

        ex_x = arrow_length * c
        ex_y = arrow_length * s
        ey_x = arrow_length * -s
        ey_y = arrow_length * c

        # x-axis (red)
        ax.plot(
            [float(xi), float(xi + ex_x)],
            [float(yi), float(yi + ex_y)],
            color=(1, 0, 0, float(a)),
            linewidth=1.5,
        )
        # y-axis (green)
        ax.plot(
            [float(xi), float(xi + ey_x)],
            [float(yi), float(yi + ey_y)],
            color=(0, 1, 0, float(a)),
            linewidth=1.5,
        )

    # Reference pose (if any)
    if ref_pose is not None:
        ref_x = -ref_pose[1]
        ref_y = ref_pose[0]
        qw = ref_pose[3]
        qx = ref_pose[4]
        qy = ref_pose[5]
        qz = ref_pose[6]
        ref_yaw = quat_to_yaw(qx, qy, qz, qw)

        c = float(jnp.cos(ref_yaw))
        s = float(jnp.sin(ref_yaw))
        ex_x = arrow_length * c
        ex_y = arrow_length * s
        ey_x = arrow_length * -s
        ey_y = arrow_length * c

        ax.plot(
            [float(ref_x), float(ref_x + ex_x)],
            [float(ref_y), float(ref_y + ex_y)],
            color=(0, 0, 0, 1.0),
            linewidth=2.0,
        )
        ax.plot(
            [float(ref_x), float(ref_x + ey_x)],
            [float(ref_y), float(ref_y + ey_y)],
            color=(0, 0, 0, 1.0),
            linewidth=2.0,
        )

    # Axes limits etc.
    ax.set_ylim(0.2, 0.8)
    ax.set_xlim(-0.35, 0.35)
    ax.set_aspect("equal", adjustable="box")

    return ax

