from abc import ABC, abstractmethod
from typing import Tuple , List

import jax
import jax.numpy as jnp
import mujoco
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
                 randomized_joints: dict,
                 num_randomizations: int):
        self.task = task
        self.controller = controller
        self.rng = np.random.RandomState(seed)
        self.num_randomizations = num_randomizations
        # structure
        # {field_name: (min, max, idxs_to_randomize)}
        self.randomized_fields = randomized_fields
        self.randomized_joints = randomized_joints
        self.model = controller.model

    def _dof_ids_from_joint_id(self, joint_id: int):
        s = self.model.jnt_dofadr[joint_id]
        joint_type = self.model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            n = 3
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            n = 6
        else:
            n = 1
        return range(s, s + n)

    def dof_ids(self, joint_names: List[str]) -> range:
        joint_ids = [mujoco.mj_name2id(self.task.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        
        dof_ids = []
        for jid in joint_ids:
            dof_ids.extend(self._dof_ids_from_joint_id(jid))
        return dof_ids
    
    
    def set_damping(self, joint_names: List[str], c: np.ndarray):

        # c has shape (num_randomizations, num_joints)
        if c.shape[1] != len(joint_names):
            raise ValueError("Damping array shape does not match number of joints.")
        if c.shape[0] != self.num_randomizations:
            raise ValueError("Damping array shape does not match number of randomizations.")

        dof_ids = self.dof_ids(joint_names)

        current_damping = self.model.dof_damping
        if current_damping.ndim == 1:
            current_damping = jnp.tile(current_damping, (self.num_randomizations,1))
        new_damping = current_damping.at[:, dof_ids].set(c)
        return new_damping

        
    def get_current_randomizations(self) -> dict:
        """Get the current randomizations in model structure.

        Returns:
            A dictionary mapping field names to their current randomized values.
        """
        current_randomizations = {}
        for field in self.randomized_fields.keys():
            model_field = getattr(self.controller.model, field)
            current_randomizations[field] = model_field[:,self.randomized_fields[field][2]]
        for field in self.randomized_joints.keys():
            joint_names = self.randomized_joints[field][2]
            
            dof_ids = self.dof_ids(joint_names)
            model_field = getattr(self.controller.model, field)
            if model_field.ndim == 1:
                model_field = jnp.tile(model_field, (self.num_randomizations,1))
            current_randomizations[field] = model_field[:, dof_ids]
        return current_randomizations
    
    def _set_randomizations(self, randomizations: dict):
        """Map randomizations to model structure.
        
            Returns a dict mapping field names to jnp.ndarray with shape matching model fields.
        """
        new_randomizations = {}
        for field in randomizations.keys():
            if field == "dof_damping":
                # need to be careful with dofs
                values = self.set_damping(self.randomized_joints["dof_damping"][2], randomizations[field])
                new_randomizations[field] = values
            else:
                values = randomizations[field] #* np.random.uniform(0.9, 1.1, size=(self.num_randomizations,))
                model_field = getattr(self.controller.model, field)        
                new_randomizations[field] = model_field.at[:, self.randomized_fields[field][2]].set(values)
              
        return new_randomizations

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
                 randomized_joints: dict,
                 num_randomizations: int):
        super().__init__(seed, task, controller, randomized_fields, randomized_joints,num_randomizations)

    def _compute_randomizations(self, signal: np.ndarray) -> jnp.ndarray:
    
        randomizations = {}
        for field, (min_val, max_val, idxs_to_randomize) in self.randomized_fields.items():
            randomizations[field] = jnp.linspace(min_val, max_val, self.num_randomizations)
        return randomizations
    


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
                 randomized_joints: dict,
                 num_randomizations: int,
                 mutation_rate: float = 0.1, 
                 elite_fraction: float = 0.5):
        super().__init__(seed, task, controller, randomized_fields, randomized_joints, num_randomizations)
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
            
            # model_field = getattr(self.controller.model, field)
            new_randomizations[field] = new_values #model_field.at[:, self.randomized_fields[field][2]].set(new_values)
        for field, (min_val, max_val, joint_names) in self.randomized_joints.items():
            elite_values = randomizations[field][elite_indices]
            # Generate new values by mutating elites
            new_values = np.empty_like(randomizations[field])
            
            # mutate elites
            for idx, ev in zip(range(self.num_elites), elite_values):
                mutated_value = self._mutate(ev, min_val, max_val)
                new_values[idx, ...] = mutated_value

            # sample globally for the rest
            sampled_value = np.random.uniform(min_val, max_val, size=(self.num_randomizations - self.num_elites, *new_values.shape[1:]))
            new_values[self.num_elites:, ...] = sampled_value
            
            dof_ids = self.dof_ids(joint_names)
            # model_field = geattr(self.controller.model, field)
            new_randomizations[field] = new_values #model_field.at[:, dof_ids].set(new_values)

        new_randomizations = self._set_randomizations(new_randomizations)
        weights = jnp.ones(self.num_randomizations) / self.num_randomizations
        
        return tuple((True, new_randomizations, weights))