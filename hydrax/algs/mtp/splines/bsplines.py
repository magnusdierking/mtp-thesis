from functools import partial
import jax
import jax.numpy as jnp
from jax import jit


# @partial(jit, static_argnums=(1, 2))
# def compute_b_spline_matrix(x: jax.Array, degree: int, num_points: int) -> jax.Array:
#     """
#     Compute the B-spline basis matrix for a given degree and knot vector.
#     !! Skips the first and last knot values in the parameter domain. !!

#     Parameters:
#         x (list or jax.Array): The knot vector.
#         degree (int): The degree of the B-spline basis.
#         num_points (int): The number of points to sample in the parameter domain.

#     Returns:
#         jax.Array: A matrix where each row represents a parameter value and each column corresponds to a basis function.
#     """
#     t_values = jnp.linspace(x[1], x[-2], num_points + 4)[2:-2]  # Exclude the first and last knot values
#     # -----------------------------------------------------------------
#     # Step 1: Initialize with degree-0 (piecewise constant) basis funcs:
#     # N_{i,0}(t) = 1 if x_i <= t < x_{i+1}, else 0.
#     # This gives a matrix of shape (M, num_basis_0).
#     # -----------------------------------------------------------------
#     b = jnp.where(
#         (x[:-1] <= t_values[:, None]) & (t_values[:, None] <= x[1:]),
#         1.0, 0.0
#     )
    
#     # -----------------------------------------------------------------
#     # Step 2: Build higher degree basis functions recursively.
#     # Cox–de Boor recursion:
#     #
#     # N_{i,d}(t) =
#     #   (t - x_i) / (x_{i+d} - x_i)     * N_{i,d-1}(t)
#     # + (x_{i+d+1} - t) / (x_{i+d+1} - x_{i+1}) * N_{i+1,d-1}(t)
#     #
#     # We update b in-place: at each loop, b contains the N_{i,d}(t).
#     # -----------------------------------------------------------------

#     for d in range(1, degree + 1):
#         # Left term:  ((t - x_i)/(x_{i+d} - x_i)) * N_{i,d-1}(t)
#         left_d1, left_d2 = x[d:-1], x[:-d - 1]  # t_{i+d} - t_i
#         b_left = jnp.where(left_d1 > left_d2, ((t_values[:, None] - left_d2) / (left_d1 - left_d2)) * b[:, :-1], 0.0)
#         # Right term: ((x_{i+d+1} - t)/(x_{i+d+1} - x_{i+1})) * N_{i+1,d-1}(t)
#         right_d1, right_d2 = x[d + 1:], x[1:-d]  # t_{i+d+1} - t_{i+1}
#         b_right = jnp.where(right_d1 > right_d2, ((right_d1 - t_values[:, None]) / (right_d1 - right_d2)) * b[:, 1:], 0.0)
#         b = b_left + b_right

#     # -----------------------------------------------------------------
#     # After the loop:
#     # b[j,i] = N_{i,degree}(t_j)
#     # Shape: (len(t_values), num_basis_functions)
#     # -----------------------------------------------------------------
#     return b


# @partial(jit, static_argnums=(1, 2, 3))
# def sample_with_replacement(rng: jax.Array, M: int, N: int, num_samples: int) -> jax.Array:
#     return jax.random.randint(rng, (num_samples, M), 0, N)


 
@partial(jit, static_argnums=(1, 2, 3))
def compute_b_spline_matrix(x: jax.Array, degree: int, num_points: int, dtype = jnp.float32) -> jax.Array:
    """
     Compute the B-spline basis matrix for a given degree and knot vector.
 
     Parameters:
         x (list or jax.Array): The knot vector.
         degree (int): The degree of the B-spline basis.
         num_points (int): The number of points to sample in the parameter domain.
 
     Returns:
         jax.Array: A matrix where each row represents a parameter value and each column corresponds to a basis function.
    """
    x = jnp.asarray(x, dtype=dtype)
    # t_values = jnp.linspace(x[0], x[-1], num_points + 4, dtype=dtype)
    t_values = jnp.linspace(x[degree], x[-1-degree], num_points + 1, dtype=dtype)[1:]
     # -----------------------------------------------------------------
     # Step 1: Initialize with degree-0 (piecewise constant) basis funcs:
     # N_{i,0}(t) = 1 if x_i <= t < x_{i+1}, else 0.
     # This gives a matrix of shape (M, num_basis_0).
     # -----------------------------------------------------------------
    one = jnp.array(1.0, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)
    b = jnp.where(
         (x[:-1] <= t_values[:, None]) & (t_values[:, None] < x[1:]),
        one,
        zero,
    )
     
     # -----------------------------------------------------------------
     # Step 2: Build higher degree basis functions recursively.
     # Cox–de Boor recursion:
     #
     # N_{i,d}(t) =
     #   (t - x_i) / (x_{i+d} - x_i)     * N_{i,d-1}(t)
     # + (x_{i+d+1} - t) / (x_{i+d+1} - x_{i+1}) * N_{i+1,d-1}(t)
     #
     # We update b in-place: at each loop, b contains the N_{i,d}(t).
     # -----------------------------------------------------------------
 
    for d in range(1, degree + 1):
        # Left term:  ((t - x_i)/(x_{i+d} - x_i)) * N_{i,d-1}(t)
        left_d1, left_d2 = x[d:-1], x[:-d - 1]  # t_{i+d} - t_i
        b_left = jnp.where(
            left_d1 > left_d2,
            ((t_values[:, None] - left_d2) / (left_d1 - left_d2)) * b[:, :-1],
            zero,
        )
        # Right term: ((x_{i+d+1} - t)/(x_{i+d+1} - x_{i+1})) * N_{i+1,d-1}(t)
        right_d1, right_d2 = x[d + 1:], x[1:-d]  # t_{i+d+1} - t_{i+1}
        b_right = jnp.where(
            right_d1 > right_d2,
            ((right_d1 - t_values[:, None]) / (right_d1 - right_d2)) * b[:, 1:],
            zero,
        )
        b = b_left + b_right
        
    last = b.shape[0] - 1
    last_basis = b.at[last, :].set(0.0).at[last, -1].set(1.0)

    b = b.at[last, :].set(last_basis[last, :])
 
     # -----------------------------------------------------------------
     # After the loop:
     # b[j,i] = N_{i,degree}(t_j)
     # Shape: (len(t_values), num_basis_functions)
     # -----------------------------------------------------------------
    return b
 
 
@partial(jit, static_argnums=(1, 2, 3))
def sample_with_replacement(rng: jax.Array, M: int, N: int, num_samples: int) -> jax.Array:
     return jax.random.randint(rng, (num_samples, M), 0, N)
 
