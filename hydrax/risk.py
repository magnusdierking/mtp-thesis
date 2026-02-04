from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class RiskStrategy(ABC):
    """An abstract risk strategy interface.

    A risk strategy defines how we combine costs from different domains with
    domain randomization. For example, a very risk-averse strategy might take
    the worst-case cost over all randomizations, while a risk-seeking strategy
    might take the best-case cost.
    """
    def __init__(self, weights: jax.Array = None):
        """Initialize the risk strategy with optional weights."""
        if weights is not None:
            self.weights = weights 
        else:
            self.weights = None

    def set_weights(self, weights: jax.Array):
        """Set the weights for the expectation."""
        self.weights = weights 

    @abstractmethod
    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Combine costs from different randomizations.

        Args:
            costs: rollout costs, size (randomizations, samples, horizon)

        Returns:
            The combined cost, size (samples, horizon).
        """
        pass



class ExpectedCost(RiskStrategy):
    """Average cost risk strategy.

    This is the standard expectation w.r.t. a customizable measure.
    """
    def __init__(self, weights: jax.Array):
        """Set the weights for the expectation."""
        self.weights = weights / jnp.sum(weights)

    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Take the average cost over all randomizations."""
        return jnp.average(costs, axis=0, weights=self.weights)


class AverageCost(RiskStrategy):
    """Average cost risk strategy.

    This is the standard expectation over randomizations that is often used in
    reinforcement learning.
    """

    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Take the average cost over all randomizations."""
        return jnp.mean(costs, axis=0)


class WorstCase(RiskStrategy):
    """A pessimistic worst-case-cost risk strategy."""

    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Take the highest cost over all randomizations."""
        return jnp.max(costs, axis=0)


class BestCase(RiskStrategy):
    """An optimistic best-case-cost risk strategy."""

    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Take the lowest cost over all randomizations."""
        return jnp.min(costs, axis=0)


class ExponentialWeightedAverage(RiskStrategy):
    """An exponential weighted average risk strategy.

    Costs are combined using a weighted average with weights

        wᵢ = exp(γ cᵢ)/∑ⱼexp(γ cⱼ).

    The parameter γ controls the risk-aversion of the strategy: positive values
    encode a risk-averse strategy, while negative values lead to risk-seeking.
    """

    def __init__(self, gamma: float):
        """Set the risk-aversion parameter γ."""
        self.gamma = gamma

    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Combine costs using an exponential weighted average."""
        weights = jax.nn.softmax(self.gamma * costs, axis=0)
        return jnp.sum(weights * costs, axis=0)


class ValueAtRisk(RiskStrategy):
    """Take the cost value at the (1 - α) quantile."""

    def __init__(self, alpha: float):
        """Set the quantile level α."""
        self.alpha = alpha

    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Take the cost value at the (1 - α) quantile."""
        return jnp.quantile(costs, 1.0 - self.alpha, axis=0)


class ConditionalValueAtRisk(RiskStrategy):
    """Take the expected cost in the tail beyond the (1 - α) quantile."""

    def __init__(self, alpha: float, weights: jax.Array = None):
        """Set the quantile level α."""
        self.alpha = alpha
        self.weights = weights / jnp.sum(weights)
        

    def combine_costs(self, costs: jax.Array) -> jax.Array:
    
        quant = jnp.quantile(costs, 1.0 - self.alpha, axis=0)
        jax.debug.print("Shape of costs: {shape}", shape=costs.shape)
        mask = jnp.where(costs >= quant, 1.0, 0.0)
        tmp_cost = jnp.where(costs >= quant, costs, 0.0)
        nbr_values = jnp.sum(mask, axis=0)

        normalized_weights = self.weights[:, None] * mask / nbr_values
        return jnp.average(tmp_cost, axis=0, weights=normalized_weights)


class InverseValueAtRisk(RiskStrategy):
    """Take the cost value at the (1-α) quantile. (alpha fraction of cost is below this value)"""


    def __init__(self, alpha: float):
        """Set the quantile level α."""
        self.alpha = alpha

    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Take the cost value at the α quantile."""
        return jnp.quantile(costs, self.alpha, axis=0)
     

class InverseConditionalValueAtRisk(RiskStrategy):
    """Take the expected cost in the head below the (1 - α) quantile. 
    (highest alpha fraction of cost ignored)"""

    def __init__(self, alpha: float, weights: jax.Array = None):
        """Set the quantile level α."""
        self.alpha = alpha
        self.weights = weights / jnp.sum(weights)

    def combine_costs(self, costs: jax.Array) -> jax.Array:
        """Take the expected cost in the head below the (1 - α) quantile."""
        quant = jnp.quantile(costs, 1.0 - self.alpha, axis=0)
        jax.debug.print("Shape of costs: {shape}", shape=costs.shape)
        mask = jnp.where(costs <= quant, 1.0, 0.0)
        tmp_cost = jnp.where(costs <= quant, costs, 0.0)
        nbr_values = jnp.sum(mask, axis=0)

        normalized_weights = self.weights[:, None] * mask / jnp.sum(self.weights[:, None] * mask, axis=0)
        return jnp.average(tmp_cost, axis=0, weights=normalized_weights)
