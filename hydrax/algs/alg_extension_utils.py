# This module contains utility functions for extending algorithms (MPPI, CEM, MTP)
import math
import jax
import jax.numpy as jnp
from functools import partial



def _colorize_1d(noise_1d, remove_dc=True, eps=1e-8, alpha_noise=3.0):
    """
    noise_1d: (T,) white ~ N(0, 1)
    Returns colored noise with approx unit variance for this 1D trajectory.
    """
    T = noise_1d.shape[0]

    # FFT -> apply magnitude shaping -> IFFT
    Xf = jnp.fft.rfft(noise_1d, axis=0)        # (T_r,)
    freqs = jnp.fft.rfftfreq(T)                # (T_r,)
    H = (1.0 / jnp.maximum(freqs, eps)) ** (alpha_noise / 2.0)
    if remove_dc:
        H = H.at[0].set(0.0)

    Yf = Xf * H                                 # (T_r,)
    y = jnp.fft.irfft(Yf, n=T, axis=0)          # (T,)

    # de-mean and unit-std
    # y = y - y.mean(axis=0, keepdims=True)
    y = y / (y.std(axis=0, keepdims=True) + eps)

    return y


@partial(jax.jit, static_argnums=(1, 2, 3))
def colorize_time_series(noise, remove_dc=True, eps=1e-8, alpha_noise=3.0):
    """
    noise: (B, T, D) white ~ N(0, 1)
    Returns colored noise with approx unit variance per (B, D) trajectory.
    """

    # Fix hyperparameters into a unary function on 1D time series: (T,) -> (T,)
    def colorize_fixed(x):
        return _colorize_1d(x, remove_dc=remove_dc, eps=eps, alpha_noise=alpha_noise)

    # First vmap over last axis (D): (T, D) -> (T, D)
    colorize_over_D = jax.vmap(colorize_fixed, in_axes=-1, out_axes=-1)

    # Then vmap over batch axis (B): (B, T, D) -> (B, T, D)
    colorize_over_BD = jax.vmap(colorize_over_D, in_axes=0, out_axes=0)

    return colorize_over_BD(noise)



# @partial(jax.jit, static_argnums=(1, 2, 3))
# def colorize_time_series(noise, remove_dc=True, eps=1e-8, alpha_noise=3.0):
#     """
#     noise: (B, T, D) white ~ N(0,1)
#     Returns colored noise with approx unit variance per (B,D) trajectory.
#     alpha=0 -> white, 1 -> pink (1/f), 2 -> brown (1/f^2)
#     """
#     B, T, D = noise.shape

#     # reshape to (B*D, T) to FFT each series independently
#     x = noise.reshape(B * D, T)

#     # FFT -> apply magnitude shaping -> IFFT
#     Xf = jnp.fft.rfft(x, axis=-1)                           # (B*D, T_r)
#     freqs = jnp.fft.rfftfreq(T)                             # (T_r,)
#     H = (1.0 / jnp.maximum(freqs, eps)) ** (alpha_noise / 2.0)    # magnitude shaping
#     if remove_dc:
#         H = H.at[0].set(0.0)
#     Yf = Xf * H[None, :]                                    # broadcast
#     y = jnp.fft.irfft(Yf, n=T, axis=-1)                     # (B*D, T)

#     # de-mean and unit-std per series (robust for finite T)
#     y = y - y.mean(axis=-1, keepdims=True)
#     y = y / (y.std(axis=-1, keepdims=True) + eps)

#     return y.reshape(B, T, D)


@partial(jax.jit, static_argnums=(1,))
def shift_tensor(tensor: jax.Array, amount: int) -> jax.Array:
    """Shift the tensor to the left and append last value at the end along axis 0."""
    shifted = jnp.roll(tensor, -amount, axis=0)
    shifted = shifted.at[-amount:, :].set(tensor[-1, :])  # hold last value
    return shifted




def savgol_coeffs(window_length: int,
                  polyorder: int,
                  deriv: int = 0,
                  delta: float = 1.0) -> jnp.ndarray:
    """Compute 1D Savitzky–Golay coefficients."""
    if window_length % 2 != 1:
        raise ValueError("window_length must be odd")
    if window_length <= polyorder:
        raise ValueError("window_length must be > polyorder")
    if deriv < 0:
        raise ValueError("deriv must be >= 0")

    half = window_length // 2

    # positions: [-half, ..., 0, ..., +half]
    x = jnp.arange(-half, half + 1, dtype=jnp.float32)

    # Vandermonde: A[i, j] = x_i ** j
    powers = jnp.arange(polyorder + 1, dtype=jnp.float32)
    A = x[:, None] ** powers[None, :]  # (window_length, polyorder+1)

    ATA = A.T @ A
    ATA_inv = jnp.linalg.pinv(ATA)
    B = ATA_inv @ A.T  # (polyorder+1, window_length)

    scale = math.factorial(deriv) / (delta ** deriv)
    coeffs = B[deriv] * scale  # (window_length,)

    return coeffs



def make_savgol_filter(window_length: int,
                       polyorder: int,
                       deriv: int = 0,
                       delta: float = 1.0,
                       axis: int = -1):
    """
    Returns a JIT-compiled function f(x) -> filtered_x

    - x can have any shape
    - Filtering is applied along `axis`
    """
    coeffs = savgol_coeffs(window_length, polyorder, deriv, delta)
    coeffs = coeffs[::-1]  # for convolution
    half = window_length // 2

    @jax.jit
    def apply(x: jnp.ndarray) -> jnp.ndarray:
        # Normalize axis to be positive
        ax = axis if axis >= 0 else x.ndim + axis

        # Move the target axis to the last position
        x_moved = jnp.moveaxis(x, ax, -1)  # shape (..., T)

        # Build pad_width: pad only last axis (time)
        pad_width = ((0, 0),) * (x_moved.ndim - 1) + ((half, half),)
        x_pad = jnp.pad(x_moved, pad_width=pad_width, mode="reflect")

        leading_shape = x_pad.shape[:-1]
        T_pad = x_pad.shape[-1]

        flat = x_pad.reshape(-1, T_pad)  # (batch_flat, T_pad)

        def filter_1d(row):
            return jnp.convolve(row, coeffs, mode="valid")  # length T

        flat_out = jax.vmap(filter_1d, in_axes=0)(flat)  # (batch_flat, T)
        T = flat_out.shape[-1]

        out_moved = flat_out.reshape(*leading_shape, T)

        # Move axis back to original position
        out = jnp.moveaxis(out_moved, -1, ax)
        return out

    return apply