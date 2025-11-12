# This module contains utility functions for extending algorithms (MPPI, CEM, MTP)
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(1, 2, 3))
def colorize_time_series(noise, remove_dc=True, eps=1e-8, alpha_noise=3.0):
    """
    noise: (B, T, D) white ~ N(0,1)
    Returns colored noise with approx unit variance per (B,D) trajectory.
    alpha=0 -> white, 1 -> pink (1/f), 2 -> brown (1/f^2)
    """
    B, T, D = noise.shape

    # reshape to (B*D, T) to FFT each series independently
    x = noise.reshape(B * D, T)

    # FFT -> apply magnitude shaping -> IFFT
    Xf = jnp.fft.rfft(x, axis=-1)                           # (B*D, T_r)
    freqs = jnp.fft.rfftfreq(T)                             # (T_r,)
    H = (1.0 / jnp.maximum(freqs, eps)) ** (alpha_noise / 2.0)    # magnitude shaping
    if remove_dc:
        H = H.at[0].set(0.0)
    Yf = Xf * H[None, :]                                    # broadcast
    y = jnp.fft.irfft(Yf, n=T, axis=-1)                     # (B*D, T)

    # de-mean and unit-std per series (robust for finite T)
    y = y - y.mean(axis=-1, keepdims=True)
    y = y / (y.std(axis=-1, keepdims=True) + eps)

    return y.reshape(B, T, D)


@partial(jax.jit, static_argnums=(1,))
def shift_tensor(tensor: jax.Array, amount: int) -> jax.Array:
    """Shift the tensor to the left and append last value at the end along axis 0."""
    shifted = jnp.roll(tensor, -amount, axis=0)
    shifted = shifted.at[-amount:, :].set(tensor[-1, :])  # hold last value
    return shifted