from pyexpat import model
import time
from typing import Dict
import os 
import jax
import jax.scipy as jsp
from jax import lax
from jaxlie import SE3

import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from functools import partial

from hydrax.files import get_root_path
from hydrax.task_base import Task

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from hydrax.utils.utils import mujoco_to_scipy_quat, quat_normalize, quat_conj, quat_mul, quat_error_body, quat_to_rotvec

import math

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)

class PushTFranka(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(
        self, planning_horizon: int = 16, sim_steps_per_control_step: int = 5, 
        nu: int = 2, 
        ctrl_limits = {"u_min": jnp.array([-0.45, -0.45]), "u_max": jnp.array([0.45, 0.45])},
        trace_sites=["T_1", "T_2","ee_site"],
        actuation_type: str = 'velocity',
        sampling_space: str = 'velocity',
        block_type: str = 'free', # 'free' or 'joint'
        ik_type: str = 'pinv',
        det_init: dict = {},
    ):
        """Load the MuJoCo model and set task parameters."""
        self.sampling_space = sampling_space
        self.actuation_type = actuation_type
        if actuation_type == 'position':
            mj_model = mujoco.MjModel.from_xml_path(
                (get_root_path() / "models" / "fr3_pushT_pos" / "scene_mjx.xml").as_posix()
            )
        elif actuation_type == 'velocity':
            if block_type == 'joint':
                mj_model = mujoco.MjModel.from_xml_path(
                    (get_root_path() / "models" / "fr3_pushT_vel" / "scene_mjx.xml").as_posix()
                )
            elif block_type == 'free':
                mj_model = mujoco.MjModel.from_xml_path(
                    # (get_root_path() / "models" / "fr3_pushT_vel" / "scene_mjx_free_exp.xml").as_posix()
                    (get_root_path() / "models" / "fr3_pushT_vel" / "scene_mjx_free.xml").as_posix()
                )
            elif block_type == 'sim-real':
                mj_model = mujoco.MjModel.from_xml_path(
                    (get_root_path() / "models" / "fr3_pushT_vel" / "scene_mjx_sim_real.xml").as_posix()
                )
            else:
                raise ValueError("block_type must be 'joint', 'free' or 'sim-real'")
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
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position_world"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation_world"
        )
        self.goal_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "goal_position_world"
        )
        self.goal_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "goal_orientation_world"
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
        self.ee_goal_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "safety"
        )
        self.ee_t1_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_t1"
        )   
        self.ee_t2_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_t2"
        )   

         # Get block body id
        
        self.T_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.block_type = block_type

        # Get block joint indices
        if block_type == 'joint' or block_type == 'sim-real':
            self.block_joint_names = ['T_x', 'T_y', 'T_z']
        elif block_type == 'free':
            self.block_joint_names = ['T']  

       
        self.block_joint_idxs = [mj_model.joint(name).id for name in self.block_joint_names]
        
        # Get actuator joint indices
        self.actuator_joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
        self.actuator_joint_ids = [mj_model.joint(name).id for name in self.actuator_joint_names]
        self.actuator_joint_idxs = self.mj_model.jnt_qposadr[self.actuator_joint_ids]
        self.dof_adr  = self.mj_model.jnt_dofadr[self.actuator_joint_ids]

        self.joint_limits = self.mj_model.jnt_range[self.actuator_joint_ids]
        
        # special to this task
        self.ee_body_id = self.mj_model.body("ee_frame").id
        self.goal_quat_block = jnp.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        # initial end effector
        self.goal_quat_ee = jnp.array([0.0, 0.7071, 0.7071, 0.0])  # [w, x, y, z]
        self.goal_pos_ee = jnp.array([0.3, 0.0, 0.045]) 

        self.det_init = det_init

    def reset(self, seed: int = 0) -> None:
        """Randomize the initial pose of the T-shaped block."""
        # Set the random seed for reproducibility
        np.random.seed(seed)
        mj_model = self.mj_model
        mj_data = mujoco.MjData(self.mj_model)

        sign_x = np.random.choice([-1, 1])
        pos_x = self.det_init.get("block_pos_x", sign_x * np.random.uniform(low=0.1, high=0.25))
        pos_y = self.det_init.get("block_pos_y", np.random.uniform(low=-0.1, high=0.15))
        angle = self.det_init.get("block_angle", np.random.uniform(np.pi/4, np.pi))
        self.goal_pos_ee = self.det_init.get("ee_goal_pos", self.goal_pos_ee)


        # Assuming the block's pose is at the beginning of qpos
        mj_data.qpos[0] = pos_x
        mj_data.qpos[1] = pos_y
        if self.block_type == 'joint' or self.block_type == 'sim-real':
            mj_data.qpos[2] = angle
        else:
            quat = euler_to_quaternion(0, 0, angle - np.pi/2)  # roll, pitch, yaw to x ,y,z,w
            mj_data.qpos[3:7] = np.array([quat[3], quat[0], quat[1], quat[2]])  # w, x, y, z

        # # Initial guess
        q = np.array([0.0, -np.pi/4, 0.0, -9*np.pi/10, 0.0, 3*np.pi/4, np.pi/4])

        # IK loop parameters
        max_iters = 1_000
        tolerance = 1e-3
        damping = 100e-3
        step_size = 1.0

        ik_start_time = time.time()
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
                print(f"Converged in {i} iterations. Took {time.time() - ik_start_time:.4f} s")
                break

            # Compute Jacobian of the EE
            J_pos = np.zeros((3, self.mj_model.nv))
            J_rot = np.zeros((3, self.mj_model.nv))
            mujoco.mj_jacBody(self.mj_model, mj_data, J_pos, J_rot, self.ee_body_id)

            # Slice columns corresponding to actuated joints
            J = np.vstack([J_pos[:, self.dof_adr], J_rot[:, self.dof_adr]])  # shape (6, n_joints)

            # Solve damped least squares: dq = (JᵀJ + λ²I)⁻¹ Jᵀ e
            JTJ = J.T @ J
            H = JTJ + damping * np.eye(len(self.dof_adr))
            g = J.T @ err
            dq = np.linalg.solve(H, g)

            # Update joint configuration
            q += step_size * dq

            # Clamp to joint limits
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

        else:
            print(f"IK did not converge. {err}, {np.linalg.norm(err)}")

        mj_data.qpos[self.actuator_joint_idxs] = q  # Set the robot's joint positions

        # Initial control signal
        if self.actuation_type == 'position':
            mj_data.ctrl[:] = q
        elif self.actuation_type == 'velocity':
            mj_data.ctrl[:] = np.zeros_like(q)
        else:
            raise ValueError("actuation_type must be 'position' or 'velocity'")
        
        return mj_model, mj_data

    ##################################
    ##       Goal Error Terms       ##
    ##################################
    
    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        """ Get the position error of the block relative to a goal position."""
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        goal_adr = self.model.sensor_adr[self.goal_position_sensor]
        error = state.sensordata[sensor_adr : sensor_adr + 3] - state.sensordata[goal_adr : goal_adr + 3]
        return error

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        """ Get the orientation error of the block relative to a goal orientation."""
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4] # w, x, y, z
        block_quat = jnp.where(block_quat[0] < 0, -block_quat, block_quat)  # ensure w >= 0
        # jax.debug.print("Block quat: {q}", q=block_quat)
        goal_adr = self.model.sensor_adr[self.goal_orientation_sensor]
        goal_quat = state.sensordata[goal_adr : goal_adr + 4]     # w, x, y, z
        goal_quat = jnp.where(goal_quat[0] < 0, -goal_quat, goal_quat)  # ensure w >= 0
        # jax.debug.print("Goal quat: {q}", q=goal_quat)
        axis_angle_error = mjx._src.math.quat_sub(block_quat, goal_quat) # gives axis angle of relative rotation
        return jnp.linalg.norm(axis_angle_error)
            #mjx._src.math.quat_to_axis_angle((block_quat))[1]  # angle
    
    ################################## 
    ##      End Effector Terms      ##
    ##################################
    
    def _get_ee_block_distance(self, state: mjx.Data) -> jax.Array:
        """Get the distance between the end effector and the block."""    
        ee_t1_adr = self.model.sensor_adr[self.ee_t1_sensor]    
        ee_t1_pos = state.sensordata[ee_t1_adr : ee_t1_adr + 3]
        ee_t2_adr = self.model.sensor_adr[self.ee_t2_sensor]
        ee_t2_pos = state.sensordata[ee_t2_adr : ee_t2_adr + 3]
        return jnp.linalg.norm(ee_t1_pos, ord=1) + jnp.linalg.norm(ee_t2_pos, ord=1)

        # return jnp.linalg.norm(ee_t1_pos, ord=1) 

        
    # def _get_ee_orientation_err(self, state: mjx.Data) -> jax.Array:
    #     """Get the end effector orientation error."""
    #     sensor_adr = self.model.sensor_adr[self.ee_orientation_sensor]
    #     # Get the end effector orientation quaternion
    #     # Assuming the end effector orientation is given by a quaternion
    #     # in the sensor data   
    #     ee_quat = state.sensordata[sensor_adr : sensor_adr + 4]
    #     goal_quat = jnp.array([0.0, 0.7071, 0.7071, 0.0])  # Assuming goal orientation is aligned with x-axis
    #     return mjx._src.math.quat_sub(ee_quat, goal_quat)
    
    def _safety_zone_cost(self, state: mjx.Data) -> jax.Array:
        """Get a cost based on the distance between the end effector and the goal."""
        sensor_adr = self.model.sensor_adr[self.ee_goal_sensor]
        distance = state.sensordata[sensor_adr : sensor_adr + 3]
        
        distance = jnp.linalg.norm(distance)
        cost = jnp.where(distance > 0.4, 1.0, 0.0)

        # ee z pos
        # sensor_adr_ee = self.model.sensor_adr[self.ee_position_sensor]
        # ee_pos = state.sensordata[sensor_adr_ee : sensor_adr_ee + 3]
        # cost += jnp.where(ee_pos[2] > 0.045, 2.0, 0.0)
        return cost
    

    def running_cost(self, state: mjx.Data, control: jax.Array = None) -> jax.Array:
        
        # Goal error terms 
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)

        position_cost = jnp.linalg.norm(position_err)
        orientation_cost = jnp.linalg.norm(orientation_err)

        safety_cost = self._safety_zone_cost(state) 
        
        # attractor
        ee_block_distance = self._get_ee_block_distance(state)
        ee_block_distance_cost = ee_block_distance
        
        if self.block_type == 'joint' or self.block_type == 'sim-real':
            total_goal_err = 30 * position_cost + 3 * orientation_cost
            error = total_goal_err + 0.005 * ee_block_distance_cost  
        elif self.block_type == 'free':
            # Jitter
            total_goal_err = 30 * position_cost + 3 * orientation_cost
            error = total_goal_err + 0.005 * ee_block_distance_cost   
        return error # safety_cost 
                                                                              

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        if self.block_type == 'joint' or self.block_type == 'sim-real':
            return 10 * self.running_cost(state, jnp.zeros(self.model.nu))
        elif self.block_type == 'free':
            return 2 * self.running_cost(state, jnp.zeros(self.model.nu)) 

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        return {}

        
        
    def success(self, state):
        position_cost = self._get_position_err(state)
        orientation_cost = self._get_orientation_err(state)
        pos_err = jnp.sqrt(jnp.sum(jnp.square(position_cost))) 
        orn_err = jnp.sqrt(jnp.sum(jnp.square(orientation_cost))) 

        return pos_err + orn_err < 0.05




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

        # return gravity_comp_torque
        return None
    
    
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
            # zero velocity
            qvel = jnp.zeros_like(data0.qvel)
            ctrl = jnp.zeros_like(data0.ctrl)
            d = data0.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
            d = mjx.forward(model, d)
            pos = d.xpos[body_id]                    # (3,)
            # jax.debug.print("Quaternion: {q}", q=d.xquat[body_id])
            axis, angle = mjx._src.math.quat_to_axis_angle(d.xquat[body_id])
            rotvec = angle * axis                    # (3,)
            return jnp.concatenate([pos[:3], rotvec], axis=0)  # (6,)

        fk_jac = jax.jit(jax.jacrev(fk_fn))

        
        if self.actuation_type == 'position':
            return self.make_ik_mapper(type=self.ik_type, fk_jac=fk_jac)
        elif self.actuation_type == 'velocity':
            return self.make_differential_ik_mapper(type=self.ik_type, fk_jac=fk_jac)




    def make_differential_ik_mapper(self, type: str = 'pinv', fk_jac = None):        
            
    
        @jax.jit
        def diff_ik_mapper_transpose(data: mjx.Data, control_xy: jax.Array) -> jax.Array:
            """
            control_xy: shape (2,) desired (vx, vy) in task space.
            Enforces zero z and rotational motion: J_rot * dq = 0.
            """
            qpos = data.qpos
            J = fk_jac(qpos)                       
            J = J[:, self.actuator_joint_idxs]     
            twist = jnp.concatenate([control_xy, jnp.zeros(4)]) 

            dq = J.T @ twist
            
            if self.sampling_space == 'position':
                return qpos[self.actuator_joint_idxs] + self.mj_model.opt.timestep * dq
            elif self.sampling_space == 'velocity':
                return dq
            
            # @jax.jit
            # def ik_mapper_pinv(data: mjx.Data, control_xy: jax.Array) -> jax.Array:
            #     """
            #     control: shape (6,) desired (vx, vy, vz, wx, wy, wz) in task space.
            #     """
            #     qpos = data.qpos
            #     J = fk_jac(qpos)                       
            #     J = J[:, self.actuator_joint_idxs]     
            #     # Compute dq with damped pseudo-inverse
            #     twist = jnp.concatenate([control_xy, jnp.zeros(4)]) 
            #     # Apparently compiles better than inv
            #     lam = 1e-3
            #     JJt = J @ J.T
            #     L = jnp.linalg.cholesky(JJt + (lam*lam) * jnp.eye(6, dtype=J.dtype))
            #     y = jax.scipy.linalg.solve_triangular(L, twist, lower=True)
            #     z = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
            #     dq = J.T @ z
                
            #     if self.actuation_type == 'position':
            #         return qpos[self.actuator_joint_idxs] + self.mj_model.opt.timestep * dq
            #     elif self.actuation_type == 'velocity':
            #         return dq
                

        @jax.jit
        def diff_ik_mapper_pinv(
            data: mjx.Data,
            control_xy: jax.Array,        
            kp_ori: float = 10.0,
            lam: float = 1e-3,
        ) -> jax.Array:
            """
            Adds a corrective twist from pose error to the commanded planar twist.
            Keep your fk_jac and a pose function fk_pose(qpos)->(p(3,), R(3,3)).
            """
            qpos = data.qpos
            # Jacobian (6 x nv) restricted to actuated joints
            J_full = fk_jac(qpos)                      # (6, len(qpos)), here len(qpos)=16
            # jax.debug.print("Full Jacobian: {J}", J=J_full.shape) #(6, 16)
            # jax.debug.print("DOF adr: {dof}", dof=self.actuator_joint_idxs)
            J = J_full[:, self.actuator_joint_idxs]    # (6, n_act), (6, 7)
            # jax.debug.print("Actuated Jacobian: {J}", J=J.shape)
            
        
            J_lin = J[:3, :]    # (3, n_act)
            J_ang = J[3:, :]    # (3, n_act)

            # Current EE pose (replace fk_pose with your pose function)
            sensor_adr = self.model.sensor_adr[self.ee_orientation_sensor]
            ee_quat = data.sensordata[sensor_adr : sensor_adr + 4]

            sensor_adr_pos = self.model.sensor_adr[self.ee_position_sensor]
            ee_pos = data.sensordata[sensor_adr_pos : sensor_adr_pos + 3]

            goal_quat = jnp.array([0.0, 0.7071, 0.7071, 0.0])  #([0.0, 0.0, 0.7071, 0.7071]) # Assuming goal orientation is aligned with x-axis
            e_rot = quat_error_body(goal_quat, ee_quat)                                   # (3,)

            # Commanded planar twist + corrective twist
            twist_cmd = jnp.concatenate([control_xy, jnp.zeros(4)])     # [vx, vy, 0, 0, 0, 0]
            temp = jnp.concatenate([control_xy, jnp.array([0.045-ee_pos[2]])]) #!
            # temp = jnp.concatenate([control_xy, jnp.array([0.0])])
            twist_err = jnp.concatenate([temp, e_rot])                 # [ex, ey, ez, ewx, ewy, ewz]
            twist = twist_cmd # twist_err


            # rotational correction via nullspace
            N = jnp.eye(J.shape[1]) - jnp.linalg.pinv(J) @ J
            # twist = twist_cmd + N @ (kp_ori * J_ang.T @ e_rot)
            qnow = qpos[jnp.array(self.actuator_joint_idxs )]
            qhome = jnp.array([ 0.51199203,  0.1014329,  -0.36340348, -2.9813132,   0.50339095,  3.06692214, -1.92271156])

            dq = jnp.linalg.pinv(J) @ twist_err #+ N @ (kp_ori * (qhome - qnow))

            if self.sampling_space == 'position':
                return qpos[self.actuator_joint_idxs] + self.mj_model.opt.timestep * dq
            elif self.sampling_space == 'velocity':
                return dq 


        if type == 'transpose':
            return diff_ik_mapper_transpose
        elif type == 'pinv':
            return diff_ik_mapper_pinv


    def make_ik_mapper(self, type: str = 'pinv', fk_jac = None):    
        """
        IK mapping function from task to joint space, 
        JIT friendly to be used inside optimize
        """

        
        @jax.jit
        def ik_mapper_transpose(data: mjx.Data, desired_pose: jax.Array) -> jax.Array:
            """
            desired_pose: shape (6,) desired (x, y, z, rotvec(3)) in task space.
            """
            qpos = data.qpos
            J = fk_jac(qpos)                       
            J = J[:, self.actuator_joint_idxs]     
            current_pose = fk_jac(qpos)           
            pose_err = desired_pose - current_pose  
            dq = J.T @ pose_err
            
            if self.sampling_space == 'position':
                return qpos[self.actuator_joint_idxs] + self.mj_model.opt.timestep * dq
            elif self.sampling_space == 'velocity':
                return dq
        @jax.jit
        def ik_mapper_pinv(
            data,
            desired_xy: jnp.ndarray, #x,y 
            # actuator_joint_idxs: jnp.ndarray,
            num_iters: int = 20,
            step_size: float = 1.0,
            damping: float = 1e-3,
        ) -> jnp.ndarray:
            
            goal_quat = jnp.array([0.0, 0.7071, 0.7071, 0.0])  # wxyz
            desired_pose = SE3(wxyz_xyz=jnp.concatenate([goal_quat, jnp.array([desired_xy[0], desired_xy[1], 0.03])]))
            qpos0 = data.qpos  # treat as immutable; we just read the initial state

            def body_fn(_i, qpos, data):
                # Jacobian restricted to actuated joints
                J = fk_jac(qpos)[:, self.actuator_joint_idxs]          # (6, n_act)

                # Current task-space pose and error
                # current_pose = fk_pose(qpos)                      # (6,)
                # forward kinematics function
                data = data.replace(qpos=qpos)
                data = mjx.forward(self.model, data)
                current_pose = SE3(wxyz_xyz=jnp.concatenate([data.xquat[self.ee_body_id], data.xpos[self.ee_body_id]]))

                delta_pose = desired_pose.multiply(current_pose.inverse())
                pose_err = delta_pose.log()                       # (6,)  (linear, angular)
                
                # Damped least-squares (via Cholesky on JJ^T + λ^2 I)
                JJt = J @ J.T                                     # (6, 6)
                L = jnp.linalg.cholesky(JJt + (damping**2) * jnp.eye(6, dtype=J.dtype))
                y = jsp.linalg.solve_triangular(L, pose_err, lower=True)
                z = jsp.linalg.solve_triangular(L.T, y, lower=False)
                dq_act = J.T @ z                                  # (n_act,)

                # Scatter into full q-space and take a step
                dq_full = jnp.zeros_like(qpos).at[jnp.array(self.actuator_joint_idxs)].set(dq_act)
                qpos_next = qpos + step_size * dq_full
                return qpos_next
            

            qpos_final = lax.fori_loop(0, num_iters, partial(body_fn, data=data), qpos0)
            return qpos_final[jnp.array(self.actuator_joint_idxs)]
                    
        if type == 'transpose':
            return ik_mapper_transpose
        elif type == 'pinv':
            return ik_mapper_pinv