from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Sequence
import numpy as np
import math


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


class BetaScheduler(ABC):
    """
    Base class for beta schedulers.

    Usage:
        sched = RatioEMAScheduler(alpha=0.05).init(x0)
        beta = sched.update(x_t)  # -> float in [beta_min, beta_max]
    """
    def __init__(self, beta_min: float = 0.05, beta_max: float = 0.85, beta0: float = 0.5):
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta = _clip(beta0, self.beta_min, self.beta_max)
        self.dim: Optional[int] = None
        self._w: Optional[np.ndarray] = None  # optional per-dim weights

    def init(self, x0: Sequence[float], weights: Optional[Sequence[float]] = None) -> "BetaScheduler":
        x0 = np.asarray(x0, dtype=float)
        self.dim = int(x0.size)
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.size != self.dim:
                raise ValueError(f"weights size {w.size} != state dim {self.dim}")
            self._w = w
        self._init_impl(x0)
        return self

    def reset(self, x0: Sequence[float]) -> None:
        """Re-initialize internal state (keeps same hyperparameters)."""
        self.init(x0, weights=self._w)

    def update(self, x: Sequence[float]) -> float:
        if self.dim is None:
            raise RuntimeError("Call .init(x0) before .update(x).")
        x = np.asarray(x, dtype=float)
        if x.size != self.dim:
            raise ValueError(f"state dim changed: got {x.size}, expected {self.dim}")
        beta = float(self._update_impl(x))
        print(f"Raw beta: {beta:.3f}")
        self.beta = _clip(beta, self.beta_min, self.beta_max)
        return self.beta

    # ----- hooks for subclasses -----
    @abstractmethod
    def _init_impl(self, x0: np.ndarray) -> None:
        pass

    @abstractmethod
    def _update_impl(self, x: np.ndarray) -> float:
        pass

    # ----- helper -----
    def _norm(self, v: np.ndarray) -> float:
        """L2 norm with optional per-dim weights (whitening)."""
        if self._w is not None:
            v = self._w * v
        return float(np.linalg.norm(v) + 1e-12)


# ------------------------------
# Concrete strategies
# ------------------------------


class RatioEMAScheduler(BetaScheduler):
    """
    One-knob, parameter-free mapping with a single EMA step `alpha`.

    - Tracks EMA of x and of deviation magnitude.
    - beta_raw = 1 - dev / (dev + dev_ema)
    - beta <- (1-alpha)*beta + alpha*beta_raw
    """
    def __init__(self, alpha: float = 0.05, **kw):
        super().__init__(beta_min=kw.get("beta_min", 0.05),
                         beta_max=kw.get("beta_max", 0.85),
                         beta0=kw.get("beta0", 0.5))
        self.alpha = float(alpha)

    def _init_impl(self, x0: np.ndarray) -> None:
        self._x_ema = x0.copy()
        self._dev_ema = 1.0  # scale bootstrap

    def _update_impl(self, x: np.ndarray) -> float:
        alpha = self.alpha
        dev = self._norm(x - self._x_ema)

        # update EMAs
        self._x_ema = (1 - alpha) * self._x_ema + alpha * x
        self._dev_ema = (1 - alpha) * self._dev_ema + alpha * dev

        # parameter-free ratio in (0,1)
        beta_raw = 1.0 - dev / (dev + self._dev_ema + 1e-12)
        beta = (1 - alpha) * self.beta + alpha * beta_raw
        return beta


class LogisticEMAScheduler(BetaScheduler):
    """
    Single-alpha logistic mapping:
      z = dev / dev_ema
      beta_raw = 1 / (1 + exp(b * (z - a)))  (a: midpoint, b: steepness)
      beta <- (1-alpha)*beta + alpha*beta_raw
    """
    def __init__(self, alpha: float = 0.05, a: float = 1.0, b: float = 4.0, **kw):
        super().__init__(beta_min=kw.get("beta_min", 0.05),
                         beta_max=kw.get("beta_max", 0.85),
                         beta0=kw.get("beta0", 0.5))
        self.alpha = float(alpha)
        self.a = float(a)
        self.b = float(b)

    def _init_impl(self, x0: np.ndarray) -> None:
        self._x_ema = x0.copy()
        self._dev_ema = 1.0

    def _update_impl(self, x: np.ndarray) -> float:
        alpha = self.alpha
        dev = self._norm(x - self._x_ema)

        # EMAs
        self._x_ema = (1 - alpha) * self._x_ema + alpha * x
        self._dev_ema = (1 - alpha) * self._dev_ema + alpha * dev

        z = dev / (self._dev_ema + 1e-12)
        beta_raw = 1.0 / (1.0 + math.exp(self.b * (z - self.a)))
        beta = (1 - alpha) * self.beta + alpha * beta_raw
        return beta


class VelocityEMAScheduler(BetaScheduler):
    """
    First-difference (velocity) scheme with one alpha.

    - Tracks x_prev and EMA of |dx|.
    - beta_raw = 1 - |dx| / (|dx| + vel_ema)
    """
    def __init__(self, alpha: float = 0.05, **kw):
        super().__init__(beta_min=kw.get("beta_min", 0.05),
                         beta_max=kw.get("beta_max", 0.85),
                         beta0=kw.get("beta0", 0.5))
        self.alpha = float(alpha)

    def _init_impl(self, x0: np.ndarray) -> None:
        self._x_prev = x0.copy()
        self._vel_ema = 1.0

    def _update_impl(self, x: np.ndarray) -> float:
        alpha = self.alpha
        dx = x - self._x_prev
        vmag = self._norm(dx)
        self._x_prev = x.copy()

        self._vel_ema = (1 - alpha) * self._vel_ema + alpha * vmag

        beta_raw = 1.0 - vmag / (vmag + self._vel_ema + 1e-12)
        beta = (1 - alpha) * self.beta + alpha * beta_raw
        return beta


class AccelEMAScheduler(BetaScheduler):
    """
    Second-difference (acceleration) scheme with one alpha.

    - Tracks previous first-difference and EMA of |acc|.
    - More sensitive to rapid changes in direction.
    """
    def __init__(self, alpha: float = 0.05, **kw):
        super().__init__(beta_min=kw.get("beta_min", 0.05),
                         beta_max=kw.get("beta_max", 0.85),
                         beta0=kw.get("beta0", 0.5))
        self.alpha = float(alpha)

    def _init_impl(self, x0: np.ndarray) -> None:
        self._x_prev = x0.copy()
        self._d1_prev = np.zeros_like(x0, dtype=float)
        self._acc_ema = 1.0

    def _update_impl(self, x: np.ndarray) -> float:
        alpha = self.alpha
        d1 = x - self._x_prev
        d2 = d1 - self._d1_prev
        self._x_prev = x.copy()
        self._d1_prev = d1

        acc = self._norm(d2)
        self._acc_ema = (1 - alpha) * self._acc_ema + alpha * acc

        beta_raw = 1.0 - acc / (acc + self._acc_ema + 1e-12)
        beta = (1 - alpha) * self.beta + alpha * beta_raw
        return beta
