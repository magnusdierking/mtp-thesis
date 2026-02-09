import jax.numpy as jnp
import mujoco
from mujoco import mjx
import jax
from pytest import param



def compute_randomizations(
    batched_model: mujoco.MjModel,
    mj_model: mujoco.MjModel,
    randomization_specs: dict,
) -> mjx.Model:
    randomized_axes = []
    for type, type_dict in randomization_specs.items():
        
        for name, param_dict in type_dict.items():
            if type == 'joints':
                id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            elif type == 'bodies':
                id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            elif type == 'geoms':
                id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            else:
                raise ValueError(f"Unknown type '{type}' in randomization specs")
            for field_name in param_dict:
                if hasattr(batched_model, field_name):
                    local_idx, randomization_values = param_dict[field_name] 
                    field_array = getattr(batched_model, field_name)
                    nbr_randomizations = randomization_values.shape[0] 
                    
                    if field_array.shape[0] != nbr_randomizations:
                        reps = (nbr_randomizations,) + (1,) * field_array.ndim
                        field_array = jnp.tile(field_array, reps)
                    if local_idx is None:
                        field_array = field_array.at[:, [id]].set(jnp.expand_dims(randomization_values, axis=-1))
                    else:
                        field_array = field_array.at[:, id, [local_idx]].set(jnp.expand_dims(randomization_values, axis=-1))


                    # randomizations[field_name] = field_array
                    batched_model = batched_model.replace(**{field_name: field_array})
                    print(f"Setting type  '{type}' field '{field_name}' for {name} index {id}")
                    randomized_axes.append(field_name)
                else:
                    raise ValueError(f"{type.capitalize()} {name} has no attribute '{field_name}'")
    
    
    return batched_model, randomized_axes              
                    
                