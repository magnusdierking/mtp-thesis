from os import stat
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit

@partial(jit, static_argnums=(1,))
def interpolate_linear(path: jax.Array, num_points: int) -> jax.Array:
    """Vectorized piecewise-linear interpolation along the first axis.
    path: (M+1, U)
    returns: (num_points*(M), U)  [caller handles tail fill]
    """
    start, goal = path[:-1], path[1:]
    linspace = lambda x, y: jnp.linspace(x, y, num_points + 1)[1:]
    return jax.vmap(linspace, in_axes=(0, 0))(start, goal).reshape(-1, path.shape[-1])

# @jax.jit
# def interpolate_linear(path: jax.Array, num_points: int) -> jax.Array:
#     """
#     Vectorized piecewise-linear interpolation along the first axis.
#     path: (M+1, U)
#     returns: (M * num_points, U)   [caller can append the final waypoint if needed]
#     """
#     start = path[:-1]                  # (M, U)
#     delta = path[1:] - start           # (M, U)

#     # n points per segment, exclude the segment endpoint (so caller can "tail fill")
#     t = jnp.linspace(0.0, 1.0, num_points, endpoint=False)[:, None]   # (n, 1)

#     # Broadcast: (M, 1, U) + (1, n, 1) * (M, 1, U) --> (M, n, U)
#     segs = start[:, None, :] + t[None, :, :] * delta[:, None, :]

#     return segs.reshape(-1, path.shape[-1])   # (M*n, U)