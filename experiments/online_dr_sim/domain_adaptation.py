from abc import ABC, abstractmethod
from typing import Tuple , List

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from scipy.stats import multivariate_normal

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
                 randomized_bodies: dict,
                 randomized_joints: dict,
                 num_randomizations: int):
        self.task = task
        self.controller = controller
        self.rng = np.random.RandomState(seed)

        self.randomized_bodies = randomized_bodies
        # {body: {"field": str,
        #         "min": float, 
        #         "max": float,
        # }}
        self.randomized_joints = randomized_joints
        # {joint: {"field": str,
        #          "min": float,
        #          "max": float,}
        self.randomized_model = controller.model # model with potential multiple randomizations

        self.num_randomizations = num_randomizations
        # {field: Array,}
        self.randomized_idxs = self._convert_randomizations_to_idxs(self.randomized_bodies, self.randomized_joints)
        # uniform init
        self.current_randomizations = self.get_uniform_randomizations()
        
        
    def _convert_randomizations_to_idxs(self, randomized_bodies: dict, randomized_joints: dict):
        randomized_fields = {}
        for body_name, body_info in randomized_bodies.items():
            body_id = mujoco.mj_name2id(self.task.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                raise ValueError(f"Body name {body_name} not found in model.")
            field = body_info["field"]
            if field not in ["body_mass"]:
                print(f"Warning: Randomization of field {field} not implemented yet.")
            min_val = body_info["min"]
            max_val = body_info["max"]
            
            # check if field exists in randomized_fields    
            if field not in randomized_fields:
                randomized_fields[field] = {}
                randomized_fields[field]["idxs"] = [body_id]
                randomized_fields[field]["min"] = [min_val]
                randomized_fields[field]["max"] = [max_val]
            else:
                randomized_fields[field]["idxs"].append(body_id)
                randomized_fields[field]["min"].append(min_val)
                randomized_fields[field]["max"].append(max_val)
                
        for joint_name, joint_info in randomized_joints.items():
            joint_id = mujoco.mj_name2id(self.task.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_idxs = self._dof_ids_from_joint_id(joint_id)
            if joint_id == -1:
                raise ValueError(f"Joint name {joint_name} not found in model.")
            field = joint_info["field"]
            min_val = joint_info["min"]
            max_val = joint_info["max"]
            
            # check if field exists in randomized_fields    
            if field not in randomized_fields:
                randomized_fields[field] = {}
                randomized_fields[field]["idxs"] = joint_idxs
                randomized_fields[field]["min"] = [min_val]
                randomized_fields[field]["max"] = [max_val]
            else:
                randomized_fields[field]["idxs"].extend(joint_idxs)
                randomized_fields[field]["min"].append(min_val)
                randomized_fields[field]["max"].append(max_val)
                
        return randomized_fields 
    
    
    def get_uniform_randomizations(self) -> dict:
        """Get uniformly spaced randomizations for each field.
        Returns:
            A dictionary mapping field names to jnp.ndarray of shape (num_randomizations,).
        """
        randomizations = {}
        for field, info in self.randomized_idxs.items():
            min_vals = jnp.array(info["min"])
            max_vals = jnp.array(info["max"])
            model_field = getattr(self.controller.model, field)
            if model_field.ndim == 1:
                model_field = jnp.tile(model_field, (self.num_randomizations,1))
                
            random_vals = jnp.arange(self.num_randomizations).reshape(-1,1) / (self.num_randomizations - 1) * (max_vals - min_vals) + min_vals
            new_model_field = model_field.at[:, jnp.array(info["idxs"])].set(random_vals)
            randomizations[field] = new_model_field

        return randomizations

    def _dof_ids_from_joint_id(self, joint_id: int):
        s = self.randomized_model.jnt_dofadr[joint_id]
        joint_type = self.randomized_model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            n = 3
        elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
            n = 6
        else:
            n = 1
        return list(range(s, s + n)) # return list of dof ids for the joint

    def dof_ids(self, joint_names: List[str]) -> range:
        joint_ids = [mujoco.mj_name2id(self.task.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        
        dof_ids = []
        for jid in joint_ids:
            dof_ids.extend(self._dof_ids_from_joint_id(jid))
        return dof_ids
    
    def _update_randomization_model(self, dr_array: np.ndarray):
        """"Update self.current_randomizations based on dr_array and self.randomized_idxs."""
        current_index = 0
        for field, info in self.randomized_idxs.items():
            idxs = info["idxs"] 
            idxs = [i for i in range(current_index, current_index + len(idxs))]
            
            new_field_values = dr_array[:, idxs]  # shape (num_randomizations, num_field_params)
            model_field = getattr(self.controller.model, field)
            if model_field.ndim == 1:
                model_field = jnp.tile(model_field, (self.num_randomizations,1))
            new_model_field = model_field.at[:, jnp.array(idxs)].set(new_field_values)
            self.current_randomizations[field] = new_model_field
            current_index += len(info["idxs"])


        
    def get_current_randomizations(self) -> dict:
        """Get the current randomizations in model structure.

        Returns:
            A dictionary mapping field names to their current randomized values.
        """         
        return self.current_randomizations



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
                 randomized_bodies: dict,
                 randomized_joints: dict,
                 num_randomizations: int):
        super().__init__(seed, task, controller, randomized_bodies, randomized_joints, num_randomizations)
    

    def get_updated_randomizations(self, signal: np.ndarray) -> Tuple[bool, dict, jnp.ndarray]:
        # Dummy, always return the same uniform randomizations
        return self.current_randomizations






class EvolutionaryDomainRandomization(AdaptiveDomainRandomizationStrategy):
    """Evolutionary domain randomization strategy.

    Uses an evolutionary strategy to adapt the distribution of randomized parameters based on performance.
    """
    def __init__(self, 
                 seed: int, 
                 task: Task,
                 controller: SamplingBasedController,
                 randomized_bodies: dict,
                 randomized_joints: dict,
                 num_randomizations: int,
                 mutation_rate: float = 0.05, # standard deviation of gaussian noise added to elites
                 elite_fraction: float = 0.5, # fraction of top performers to consider as elites
                 epsilon: float = 0.85        # fraction of new individuals created via mutation
                 ):
        super().__init__(seed, task, controller, randomized_bodies, randomized_joints, num_randomizations)
        
        self.num_elites = max(1, int(elite_fraction * num_randomizations))
        self.mutation_rate = mutation_rate
        self.num_mutations = int((epsilon) * num_randomizations)
        
        # extract min and max for easy clippitn
        max_bounds = []
        min_bounds = []
        for field, info in self.randomized_idxs.items():
            max_bounds.extend(info["max"])
            min_bounds.extend(info["min"])
        self.max_bounds = np.array(max_bounds)
        self.min_bounds = np.array(min_bounds)
    
    
    def _evolve(self, dr_array: dict, signal: np.ndarray) -> np.ndarray:
        """
        dr_array (num_randomizations, total_num_randomized_params)
        signal: (num_randomizations,) lower is better
        """
        # get elite indives
        elite_indices = jnp.argsort(signal)[:self.num_elites]
        elite_dr = dr_array[elite_indices, ...]  # shape (num_elites, total_num_randomized_params)
        # sample from elites (uniform) to get num_mutations
        elite_dr = elite_dr[self.rng.choice(self.num_elites, self.num_mutations, replace=True), ...]
        # add noise to elites to create new individuals
        elite_dr = elite_dr + self.mutation_rate * np.random.randn(self.num_mutations)[...,None] * jnp.ones_like(elite_dr) # same across all elites
        new_dr = np.empty_like(dr_array)
        new_dr[:self.num_mutations,...] = elite_dr
        
        # fill rest with uniform samples
        for i in range(self.num_mutations, self.num_randomizations):
            # uniform sample
            new_dr[i,...] = self.rng.uniform(self.min_bounds, self.max_bounds)
        return new_dr
    
    
    def get_updated_randomizations(self, signal: np.ndarray) -> Tuple[bool, dict, jnp.ndarray]:
        # check if signal is informative
        min_max_scaled = (signal - jnp.min(signal)) / (jnp.max(signal) - jnp.min(signal) + 1e-8)
        relative_variation = jnp.std(min_max_scaled) / (jnp.mean(min_max_scaled) + 1e-8)

        print("Relative variation in signal:", relative_variation)
        if relative_variation < 0.75:
            print("Signal not informative enough, skipping update.")
            weights = jnp.ones(self.num_randomizations) / self.num_randomizations
            return tuple((False, self.current_randomizations, weights))
        
        
  
        dr_list = [] 
        # extract randomized values
        for field, values in self.current_randomizations.items():
            if field not in self.randomized_idxs:
                raise ValueError(f"Field {field} not in randomized_idxs.")
            idxs = self.randomized_idxs[field]["idxs"]
            # extract from model field
            field_values = values[..., idxs]  # shape (num_randomizations, num_idxs)
            dr_list.append(field_values)
        # concatenate all randomized values
        dr_array = np.concatenate(dr_list, axis=1)  # shape (num_randomizations, total_num_randomized_params)
        
        new_dr  = self._evolve(dr_array, signal)  # shape (num_randomizations, total_num_randomized_params)
        
        self._update_randomization_model(new_dr)
        
        weights = jnp.ones(self.num_randomizations) / self.num_randomizations
        
        return tuple((True, self.current_randomizations, weights))
    
    
    
    
    
    
    
class BayesianDomainRandomization(AdaptiveDomainRandomizationStrategy):
    """Evolutionary domain randomization strategy.

    Uses an evolutionary strategy to adapt the distribution of randomized parameters based on performance.
    """
    def __init__(self, 
                 seed: int, 
                 task: Task,
                 controller: SamplingBasedController,
                 randomized_bodies: dict,
                 randomized_joints: dict,
                 num_randomizations: int,
                 epsilon: float = 0.95):
        super().__init__(seed, task, controller, randomized_bodies, randomized_joints, num_randomizations)

        self.num_gaussian_samples = int((epsilon) * num_randomizations)
        
        # extract min and max for easy clippitn
        max_bounds = []
        min_bounds = []
        for field, info in self.randomized_idxs.items():
            max_bounds.extend(info["max"])
            min_bounds.extend(info["min"])
        self.max_bounds = np.array(max_bounds)
        self.min_bounds = np.array(min_bounds)
        self.mean = (self.max_bounds + self.min_bounds) / 2
        self.cov = (self.max_bounds - self.min_bounds) / 4  # initial variance for uniform dist
    
    
    def _evolve(self, dr_array: dict, signal: np.ndarray) -> np.ndarray:
        """
        dr_array (num_randomizations, total_num_randomized_params)
        signal: (num_randomizations,) lower is better
        
        Bayesian update of mean and std based on
            signal as likelihood
            last mean as prior
        """
        likelihood = signal
        # likelihood = np.exp(-signal)  # higher likelihood for lower signal
        # likelihood = likelihood / np.sum(likelihood)  # normalize
        
        prior = multivariate_normal.pdf(dr_array, mean=self.mean, cov=self.cov)  # shape (num_randomizations,)
        prior = prior / np.sum(prior)  # normalize
        
        posterior = likelihood * prior  
        # mean and cov per element
        new_mean = np.sum(dr_array * posterior[:, None], axis=0)  # shape (total_num_randomized_params,)
        diff = dr_array - new_mean[None, :]  # shape (num_randomizations, total_num_randomized_params)
        new_cov = np.sum(likelihood[:, None] * (diff ** 2), axis=0)  # shape (total_num_randomized_params,)
        self.mean = new_mean
        self.cov = np.diag(new_cov) + 1e-6  # add small value to diagonal for numerical stability
        # sample new randomizations from updated gaussian
        new_dr = np.random.multivariate_normal(self.mean, self.cov, (self.num_gaussian_samples,))
        # clip to bounds
        new_dr = np.clip(new_dr, self.min_bounds, self.max_bounds)
        # TODO use truncated normal ?
        
    
    def get_updated_randomizations(self, signal: np.ndarray) -> Tuple[bool, dict, jnp.ndarray]:
        if np.sum(signal) != 1.0:
            print("Signal needs to be a probability distribution summing to 1, skipping update.")
            return False, self.current_randomizations, jnp.ones(self.num_randomizations) / self.num_randomizations

        dr_list = []
        # extract randomized values
        for field, values in self.current_randomizations.items():
            if field not in self.randomized_idxs:
                raise ValueError(f"Field {field} not in randomized_idxs.")
            idxs = self.randomized_idxs[field]["idxs"]
            # extract from model field
            field_values = values[..., idxs]  # shape (num_randomizations, num_idxs)
            dr_list.append(field_values)
        # concatenate all randomized values
        dr_array = np.concatenate(dr_list, axis=1)  # shape (num_randomizations, total_num_randomized_params)
        
        new_dr  = self._evolve(dr_array, signal)  # shape (num_randomizations, total_num_randomized_params)
        
        self._update_randomization_model(new_dr)
        
        # TODO new weights as likelihood under new gaussian ?
        weights = jnp.ones(self.num_randomizations) / self.num_randomizations
        
        return tuple((True, self.current_randomizations, weights))