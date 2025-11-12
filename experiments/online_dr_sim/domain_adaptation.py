from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx
import numpy as np

from alg_base_opt_dr import SamplingBasedController
from flax.struct import dataclass
from hydrax.task_base import Task


class AdaptiveDomainRandomizationStrategy(ABC):
    """An abstract adaptive domain randomization strategy interface.
    """

    def __init__(self, 
                 seed: int,
                 task: Task,
                 controller: SamplingBasedController,
                 randomized_fields: dict,
                 num_randomizations: int):
        self.task = task
        self.controller = controller
        self.rng = np.random.RandomState(seed)
        self.num_randomizations = num_randomizations
        # structure
        # {field_name: (min, max, idxs_to_randomize)}
        self.randomized_fields = randomized_fields
        
    def get_current_randomizations(self) -> dict:
        """Get the current randomizations in model structure.

        Returns:
            A dictionary mapping field names to their current randomized values.
        """
        current_randomizations = {}
        for field in self.randomized_fields.keys():
            model_field = getattr(self.controller.model, field)
            current_randomizations[field] = model_field[:,self.randomized_fields[field][2]]
        return current_randomizations

    @abstractmethod
    def get_updated_randomizations(self, signal: np.ndarray, randomizations: dict) -> Tuple[bool, dict, jnp.ndarray]:
        # signal is error signal sim to real
        pass
    
    
    
    
    
class UniformDomainRandomization(AdaptiveDomainRandomizationStrategy):
    """Uniform domain randomization strategy.

    Initializes each randomized field with uniformly spaced values within the given ranges.
    """
    def __init__(self, 
                 seed: int, 
                 task: Task,
                 controller: SamplingBasedController,
                 randomized_fields: dict,
                 num_randomizations: int):
        super().__init__(seed, task, controller, randomized_fields, num_randomizations)

    def _compute_randomizations(self, signal: np.ndarray) -> jnp.ndarray:
    
        randomizations = {}
        for field, (min_val, max_val, idxs_to_randomize) in self.randomized_fields.items():
            randomizations[field] = jnp.linspace(min_val, max_val, self.num_randomizations)
        return randomizations
    
    
    def _set_randomizations(self, randomizations: dict):
        new_randomizations = {}
        for field in randomizations.keys():
            values = randomizations[field] #* np.random.uniform(0.9, 1.1, size=(self.num_randomizations,))
            model_field = getattr(self.controller.model, field)
            # def _set(value):
            #     return model_field.at[self.randomized_fields[field][2]].set(value)
            
            # new_randomizations[field] = jax.vmap(_set)(values)
            new_randomizations[field] = model_field.at[:, self.randomized_fields[field][2]].set(values)
            # print("New randomizations field after vmap shape:", new_randomizations[field].shape)
              
        return new_randomizations


    def get_updated_randomizations(self, signal: np.ndarray, randomizations: dict) -> Tuple[bool, dict, jnp.ndarray]:
        # Dummy update, always equal spacing
        # compute values
        new_randomizations = self._compute_randomizations(signal)
        # {key: jnp.ndarray(num_randomizations,)}
        # print("Computed new randomizations:", new_randomizations)
        # map to model structure
        new_randomizations = self._set_randomizations(new_randomizations)
        # print("Mapped new randomizations to model structure:", new_randomizations)
        weights = jnp.ones(self.num_randomizations) / self.num_randomizations
        
        return tuple((True, new_randomizations, weights))


class EvolutionaryDomainRandomization(AdaptiveDomainRandomizationStrategy):
    """Evolutionary domain randomization strategy.

    Uses an evolutionary strategy to adapt the distribution of randomized parameters based on performance.
    """
    def __init__(self, 
                 seed: int, 
                 task: Task,
                 controller: SamplingBasedController,
                 randomized_fields: dict,
                 num_randomizations: int,
                 mutation_rate: float = 0.1, 
                 elite_fraction: float = 0.5):
        super().__init__(seed, task, controller, randomized_fields, num_randomizations)
        self.num_elites = max(1, int(elite_fraction * num_randomizations))
        self.mutation_rate = mutation_rate
        
        
    def _mutate(self, values: jnp.ndarray, min_val: float, max_val: float) -> jnp.ndarray:
        sigma = 0.304 * (max_val - min_val)  # 90% of gaussian mass inside
        mutation = self.rng.normal(0, sigma, size=values.shape)
        mutated_values = values + mutation
        # Clip to bounds
        mutated_values = jnp.clip(mutated_values, min_val, max_val)
        return mutated_values
    
    def get_updated_randomizations(self, signal: np.ndarray, randomizations: dict) -> Tuple[bool, dict, jnp.ndarray]:
        # Select elites based on signal (lower signal is better)
        elite_indices = jnp.argsort(signal)[:self.num_elites]
        
        new_randomizations = {}
        for field, (min_val, max_val, idxs_to_randomize) in self.randomized_fields.items():
            elite_values = randomizations[field][elite_indices]
            # Generate new values by mutating elites
            new_values = []
            
            # mutate elites
            for ev in elite_values:
                mutated_value = self._mutate(ev, min_val, max_val)
                new_values.append(mutated_value)

            # sample globally for the rest
            while len(new_values) < self.num_randomizations:
                sampled_value = np.random.uniform(min_val, max_val)
                new_values.append(sampled_value)            
            new_values = jnp.array(new_values[:self.num_randomizations])
            # while len(new_values) < self.num_randomizations:
            #     for ev in elite_values:
            #         mutated_value = self._mutate(ev, min_val, max_val)
            #         new_values.append(mutated_value)
            #         if len(new_values) >= self.num_randomizations:
            #             break
            # new_values = jnp.array(new_values[:self.num_randomizations])
            
            model_field = getattr(self.controller.model, field)
            new_randomizations[field] = model_field.at[:, self.randomized_fields[field][2]].set(new_values)

        weights = jnp.ones(self.num_randomizations) / self.num_randomizations
        
        return tuple((True, new_randomizations, weights))