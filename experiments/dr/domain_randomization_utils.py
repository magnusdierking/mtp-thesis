import jax.numpy as jnp
import sys
import mujoco
from mujoco import mjx
from mujoco import mjtObj
import jax
from pytest import param
from hydrax.files import get_root_path


def compute_randomizations(
    batched_model: mjx.Model,
    mj_model: mujoco.MjModel,
    randomization_specs: dict,
) -> tuple[mjx.Model, list]:
    spec = mujoco.MjSpec()
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
                    
                




def compute_randomizations_with_derived_quantities(
    batched_model: mjx.Model,
    mj_model: mujoco.MjModel,
    xml_path: str,
    randomization_specs: dict,
) -> tuple[mjx.Model, list]:
    """
    Compute body randomizations with correct derived quantities.
    Non-body randomizations use direct pytree modification.
    """
    import jax.numpy as jnp
    
    # Check if body randomizations exist
    body_specs = randomization_specs.get('bodies', {})
    if not body_specs:
        # No body randomizations, use your original fast method
        return compute_randomizations(batched_model, mj_model, randomization_specs)
    
    # Get number of randomizations from first body parameter
    first_body = next(iter(body_specs.values()))
    first_param = next(iter(first_body.values()))
    _, values = first_param
    num_randomizations = values.shape[0]
    
    # Fields to extract after recompilation
    derived_fields = [
        'body_mass',
        'body_inertia', 
        'body_ipos',
        'body_subtreemass',
        'dof_M0',
        'body_invweight0',
        'dof_invweight0',
    ]
    
    # Storage for extracted values
    compiled_field_values = {field: [] for field in derived_fields}
    
    # Compile models with body randomizations
    for i in range(num_randomizations):
        spec = mujoco.MjSpec.from_file(xml_path)
        
        # Apply body randomizations
        for body_name, param_dict in body_specs.items():
            # Get body ID from your original model
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            
            # Find the body in the spec using the compiled model's body list
            spec_body = None
            print("=" * 10)
            # Try using spec.find method if available
            try:
                spec_body = spec.find(mujoco.mjtObj.mjOBJ_BODY, body_name)
            except:
                # Fallback: search in worldbody children
                all_bodies = spec.worldbody.find_all('body')
                for body in all_bodies:
                    if body.name == body_name:
                        spec_body = body
                        break
            
            if spec_body is None:
                raise ValueError(f"Body '{body_name}' not found in spec")
            print(dir(spec_body))
            
            # Apply parameter changes
            for field_name, (local_idx, randomization_values) in param_dict.items():
                field_value = randomization_values[i]
                if field_name == 'body_mass':
                    local_name = 'mass'
                    spec.bodies[body_id].mass = float(field_value)
                elif field_name == 'body_inertia':
                    local_name = 'inertia'
                elif field_name == 'body_ipos':
                    local_name = 'ipos'
                elif field_name == 'body_subtreemass':
                    local_name = 'subtreemass'
                else:
                    raise ValueError(f"Unknown body field '{field_name}' for mapping")
                
                # if local_idx is None:
                #     setattr(spec_body, local_name, float(field_value))
                # else:
                #     field_array = getattr(spec_body, local_name)
                #     field_array[local_idx] = float(field_value)
                
                print(f"Batch {i}: Body '{body_name}' (id={body_id}) field '{field_name}' = {field_value}")
            print(spec_body.name)
            print(spec_body.mass)
            print(spec_body.inertia)
             
        # Compile and extract derived quantities
        compiled_model = spec.compile()
      
        print(compiled_model.body_mass)
        sys.exit(0)
        print(compiled_model.body_inertia[body_id])
        
        for field in derived_fields:
            compiled_field_values[field].append(getattr(compiled_model, field).copy())
        
        del spec
    
    # Stack into batched arrays
    batched_updates = {}
    randomized_axes = []
    
    for field, values in compiled_field_values.items():
        batched_updates[field] = jnp.stack([jnp.array(v) for v in values])
        randomized_axes.append(field)
    
    # Update batched model with correct derived quantities
    batched_model = batched_model.replace(**batched_updates)
    
    # Now handle geoms and joints with your original fast method
    for type_name in ['geoms', 'joints']:
        if type_name not in randomization_specs:
            continue
            
        type_dict = randomization_specs[type_name]
        for name, param_dict in type_dict.items():
            if type_name == 'joints':
                id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            else:  # geoms
                id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            
            for field_name, (local_idx, randomization_values) in param_dict.items():
                field_array = getattr(batched_model, field_name)
                nbr_randomizations = randomization_values.shape[0]
                
                if field_array.shape[0] != nbr_randomizations:
                    reps = (nbr_randomizations,) + (1,) * field_array.ndim
                    field_array = jnp.tile(field_array, reps)
                
                if local_idx is None:
                    field_array = field_array.at[:, id].set(randomization_values)
                else:
                    field_array = field_array.at[:, id, local_idx].set(randomization_values)
                
                batched_model = batched_model.replace(**{field_name: field_array})
                randomized_axes.append(field_name)
                print(f"Setting {type_name} '{name}' field '{field_name}' (id={id})")
    
    print(f"\nTotal randomized fields: {set(randomized_axes)}")
    return batched_model, randomized_axes
