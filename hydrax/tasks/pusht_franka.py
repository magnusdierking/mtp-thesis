from typing import Dict
import os 
import jax
# from jax import config
# config.update("jax_log_compiles", True)  

import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from hydrax.files import get_root_path
from hydrax.task_base import Task

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def mujoco_to_scipy_quat(q):
    return np.array([q[1], q[2], q[3], q[0]])

class PushTFranka(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(
        self, planning_horizon: int = 15, sim_steps_per_control_step: int = 15, 
        nu: int = 2, 
        ctrl_limits = {"u_min": jnp.array([-0.45, -0.45]), "u_max": jnp.array([0.45, 0.45])},
        trace_sites=["ee_site", "T_1", "T_2"],
        actuation_type: str = 'velocity',
        ik_type: str = 'pinv',
    ):
        """Load the MuJoCo model and set task parameters."""
        self.actuation_type = actuation_type
        if actuation_type == 'position':
            mj_model = mujoco.MjModel.from_xml_path(
                (get_root_path() / "models" / "fr3_pushT_pos" / "scene_mjx.xml").as_posix()
            )
        elif actuation_type == 'velocity':
            mj_model = mujoco.MjModel.from_xml_path(
                (get_root_path() / "models" / "fr3_pushT_vel" / "scene_mjx.xml").as_posix()
            )
        else:
            raise ValueError("actuation_type must be 'position' or 'velocity'")
        
        self.ik_type = ik_type
        if ik_type not in ['transpose', 'pinv']:
            raise ValueError("ik_type must be 'transpose' or 'pinv'")


        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=trace_sites,
            nu=nu,
            ctrl_limits=ctrl_limits,
        )

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )
        self.ee_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_pos"
        )
        self.ee_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_quat"
        )
        self.block_global_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position_world"
        )
        
        self.T_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "block")
        
        # Get block joint indices
        self.block_joint_names = ['T_x', 'T_y', 'T_z']
        self.block_joint_idxs = [mj_model.joint(name).id for name in self.block_joint_names]
        
        # Get actuator joint indices
        self.actuator_joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
        self.actuator_joint_idxs = [mj_model.joint(name).id for name in self.actuator_joint_names]
        
        self.joint_limits = self.mj_model.jnt_range[self.actuator_joint_idxs]
        
        # special to this task
        self.ee_body_id = self.mj_model.body("ee_frame").id
        self.goal_quat_block = jnp.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        # initial end effector
        self.goal_quat_ee = jnp.array([0.0, 0.7071, 0.7071, 0.0])  # [w, x, y, z]
        self.goal_pos_ee = jnp.array([0.3, 0.0, 0.05]) #np.array([0.3, 0.0, 0.05])

    def reset(self, seed: int = 0) -> None:
        """Randomize the initial pose of the T-shaped block."""
        # Set the random seed for reproducibility
        np.random.seed(seed)
        mj_model = self.mj_model
        mj_model.opt.timestep = 0.002
        mj_model.opt.iterations = 20 # TODO Optimize
        mj_model.opt.ls_iterations = 20 # TODO Optimize
        mj_data = mujoco.MjData(self.mj_model)
        # Randomize the block's position and orientation
        pos_x = np.random.uniform(low=-0.3, high=0.3)
        pos_y = np.random.uniform(low=-0.25, high=0.1)
        angle = np.random.uniform(-np.pi/2, np.pi/2)

        # Assuming the block's pose is at the beginning of qpos
        mj_data.qpos[0] = pos_x
        mj_data.qpos[1] = pos_y
        mj_data.qpos[2] = angle
        
        j_start = 3  # Adjust based on your model (e.g., 3 if floating base)
        n_joints = 7

        # # Initial guess
        q = np.array([0.0, -np.pi/4, 0.0, -9*np.pi/10, 0.0, 3*np.pi/4, np.pi/4])

        # IK loop parameters
        max_iters = 100
        tolerance = 1e-4
        damping = 100e-3

        for i in range(max_iters):
            # Set current joint state
            mj_data.qpos[self.actuator_joint_idxs] = q
            mujoco.mj_forward(self.mj_model, mj_data)

            # Current EE pose
            current_pos = mj_data.xpos[self.ee_body_id]
            current_quat = mj_data.xquat[self.ee_body_id]

            # Position error
            pos_err = self.goal_pos_ee - current_pos  # shape (3,)

            # Orientation error (quaternion distance -> angular velocity vector)
            r_current = R.from_quat(mujoco_to_scipy_quat(current_quat))
            r_desired = R.from_quat(mujoco_to_scipy_quat(self.goal_quat_ee))

            # Rotation needed to go from current to desired
            r_error = r_desired * r_current.inv()

            # Convert to rotation vector (axis-angle * angle)
            orn_err = r_error.as_rotvec()  # shape (3,)

            # Combined 6D task error
            err = np.concatenate([pos_err, orn_err])  # shape (6,)

            if np.linalg.norm(err) < tolerance:
                print(f"Converged in {i} iterations.")
                break

            # Compute Jacobian of the EE
            J_pos = np.zeros((3, self.mj_model.nv))
            J_rot = np.zeros((3, self.mj_model.nv))
            mujoco.mj_jacBody(self.mj_model, mj_data, J_pos, J_rot, self.ee_body_id)

            # Slice columns corresponding to actuated joints
            J = np.vstack([J_pos[:, self.actuator_joint_idxs], J_rot[:, self.actuator_joint_idxs]])  # shape (6, n_joints)

            # Solve damped least squares: dq = (JᵀJ + λ²I)⁻¹ Jᵀ e
            JTJ = J.T @ J
            H = JTJ + damping * np.eye(n_joints)
            g = J.T @ err
            dq = np.linalg.solve(H, g)

            # Update joint configuration
            q += dq

            # Clamp to joint limits
            for j in range(n_joints):
                low, high = self.joint_limits[j]
                q[j] = np.clip(q[j], low, high)

        else:
            print("IK did not converge.")
            

        mj_data.qpos[self.actuator_joint_idxs] = q  # Set the robot's joint positions

        # initial control
        # mj_data.ctrl[:] = np.zeros(mj_model.nu)
        mj_data.ctrl[:] = q
        return mj_model, mj_data

    ##################################
    ##       Goal Error Terms       ##
    ##################################
    
    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        """ Get the position error of the block relative to a goal position."""
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        """ Get the orientation error of the block relative to a goal orientation."""
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        return mjx._src.math.quat_sub(block_quat, self.goal_quat_block)
    
    
    ################################## 
    ##      End Effector Terms      ##
    ##################################
    
    def _get_ee_block_distance(self, state: mjx.Data) -> jax.Array:
        """Get the distance between the end effector and the block."""
        sensor_adr = self.model.sensor_adr[self.ee_position_sensor]
        ee_pos = state.sensordata[sensor_adr : sensor_adr + 3]
        
        sensor_adr_block = self.model.sensor_adr[self.block_global_position_sensor]
        block_pos = state.sensordata[sensor_adr_block : sensor_adr_block + 3]
        # Calculate the Euclidean distance between the end effector and the block        
        return jnp.linalg.norm(ee_pos[:2] - block_pos[:2]) # only x,y

        
    def _get_ee_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Get the end effector orientation error."""
        sensor_adr = self.model.sensor_adr[self.ee_orientation_sensor]
        # Get the end effector orientation quaternion
        # Assuming the end effector orientation is given by a quaternion
        # in the sensor data   
        ee_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([0.0, 0.7071, 0.7071, 0.0])  # Assuming goal orientation is aligned with x-axis
        return mjx._src.math.quat_sub(ee_quat, goal_quat)
    

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        
        # Goal error terms 
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        
        total_goal_err = 5 * position_cost + orientation_cost
        
        # safety based
        ee_block_distance = self._get_ee_block_distance(state)
        ee_block_distance_cost = jnp.square(ee_block_distance)
        
        # TODO velocity error for the T ?
        control_cost = jnp.sum(jnp.square(control))  # penalize large control inputs
        error = total_goal_err+ 0.25 * ee_block_distance_cost # + 0.05 * control_cost 
        
        return error 
                                                                              

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        new_mass = self.model.body_mass.at[self.T_bid].set(jax.random.uniform(rng, (), minval=0.05, maxval=1))
        return {"body_mass": new_mass}
        # n_geoms = self.model.geom_friction.shape[0]
        # multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=1.5)
        # new_frictions = self.model.geom_friction.at[:, 0].set(
        #     self.model.geom_friction[:, 0] * multiplier
        # )
        # return {"geom_friction": new_frictions}




    ################################## 
    ##            Special           ##
    ##################################
    
    
    def make_gravity_compensator(self):
        """
        Gravity compensation torque function for the robot only, 
        JIT friendly to be used inside optimize
        """

        model = self.model
        actuator_joint_idxs = self.actuator_joint_idxs
        data0 = mjx.make_data(model)

        @jax.jit
        def gravity_comp_torque(data: mjx.Data) -> jax.Array:
            d = data0.replace(qpos=data.qpos)
            d = mjx.forward(model, d)
            # Compute gravity torques
            tau_g = mjx.inverse(model, d).qfrc_bias
            return tau_g[jnp.array(actuator_joint_idxs)]

        return gravity_comp_torque
    
    
    def make_control_mapper(self):
        """
        Control mapping function from task to joint space, 
        JIT friendly to be used inside optimize
        """

        model = self.model
        body_id = self.ee_body_id

        data0 = jax.device_put(mjx.make_data(model))

        # FK -> [x, y, rotvec(3)]
        def fk_fn(qpos):
            d = data0.replace(qpos=qpos)
            d = mjx.forward(model, d)
            pos = d.xpos[body_id]                    # (3,)
            axis, angle = mjx._src.math.quat_to_axis_angle(d.xquat[body_id])
            rotvec = angle * axis                    # (3,)
            return jnp.concatenate([pos[:3], rotvec], axis=0)  # (6,)

        fk_jac = jax.jit(jax.jacrev(fk_fn))
        
        @jax.jit
        def ik_mapper_transpose(data: mjx.Data, control_xy: jax.Array) -> jax.Array:
            """
            control_xy: shape (2,) desired (vx, vy) in task space.
            Enforces zero z and rotational motion: J_rot * dq = 0.
            """
            qpos = data.qpos
            J = fk_jac(qpos)                       
            J = J[:, self.actuator_joint_idxs]     
            twist = jnp.concatenate([control_xy, jnp.zeros(4)]) 

            dq = J.T @ twist
            
            if self.actuation_type == 'position':
                return qpos[self.actuator_joint_idxs] + self.mj_model.opt.timestep * dq
            elif self.actuation_type == 'velocity':
                return dq
        
        @jax.jit
        def ik_mapper_pinv(data: mjx.Data, control_xy: jax.Array) -> jax.Array:
            """
            control: shape (6,) desired (vx, vy, vz, wx, wy, wz) in task space.
            """
            qpos = data.qpos
            J = fk_jac(qpos)                       
            J = J[:, self.actuator_joint_idxs]     
            # Compute dq with damped pseudo-inverse
            twist = jnp.concatenate([control_xy, jnp.zeros(4)]) 
            # Apparently compiles better than inv
            lam = 1e-3
            JJt = J @ J.T
            L = jnp.linalg.cholesky(JJt + (lam*lam) * jnp.eye(6, dtype=J.dtype))
            y = jax.scipy.linalg.solve_triangular(L, twist, lower=True)
            z = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
            dq = J.T @ z
            
            if self.actuation_type == 'position':
                return qpos[self.actuator_joint_idxs] + self.mj_model.opt.timestep * dq
            elif self.actuation_type == 'velocity':
                return dq

        if self.ik_type == 'transpose':
            return ik_mapper_transpose
        elif self.ik_type == 'pinv':
            return ik_mapper_pinv


