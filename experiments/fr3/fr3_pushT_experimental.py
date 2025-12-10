# Standard library
import argparse
import threading
import copy
import csv
from curses.ascii import ctrl
import os
import pickle
import time
from math import sin, cos
from pprint import pformat
from typing import Sequence
from xml.parsers.expat import model

# Third-party: JAX
import jax
jax.config.update("jax_platform_name", "gpu")
import jax.numpy as jnp

# Third-party: Scientific computing
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from mujoco import mjx

# Third-party: ROS
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_geometry_msgs import do_transform_pose
from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix

# Third-party: ROS messages
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Point

# Third-party: Visualization
import matplotlib.pyplot as plt

# Local: Hydrax
from hydrax.algs import MPPI, MTP, AnMTP
from hydrax.alg_base import SamplingBasedController
from hydrax.files import get_root_path
from hydrax.risk import ExpectedCost, AverageCost, WorstCase, BestCase, ExponentialWeightedAverage, InverseConditionalValueAtRisk, InverseValueAtRisk
from hydrax.utils.utils import mujoco_to_scipy_quat, quat_normalize, quat_conj, quat_mul, quat_error_body, quat_to_rotvec, mat2quat, se3_left_invariant_metric
from hydrax.utils.video import VideoRecorder

# Local: Project specific
from franka_panda_server import FrankaPandaServer
from pusht_franka_free import PushTFranka



def planning_loop(seed, node):
    
    ###########################
    ##      Setup Task       ##
    ###########################

    task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=12,
                    sim_steps_per_control_step=2,
                    ctrl_limits={"u_min": jnp.array([-0.4, -0.4]), 
                                 "u_max": jnp.array([0.4, 0.4])},
                    actuation_type='velocity',
                    sampling_space="velocity",
                    block_type = 'free',
                )

    ###########################
    ##      Setup Algo       ##
    ###########################
  
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of the walker task."
    )
    subparsers = parser.add_subparsers(
        dest="algorithm", help="Sampling algorithm (choose one)"
    )
    subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
    subparsers.add_parser("mtp", help="MTP")
    args = parser.parse_args()

    # Set the controller based on command-line arguments
    if args.algorithm is None: 
        args.algorithm = "mppi"  # Default to MTP
    elif args.algorithm == "mppi":
        print("Running MPPI")
        controller = MPPI(
            task,
            num_samples=512,
            noise_level=0.3,
            temperature=0.1,
            num_randomizations=2,
            seed=seed,
        )
   
    mjx_data = mjx.make_data(controller.task.model)
    policy_params = controller.init_params(seed)
    node.get_logger().info(
            f"Planning with {controller.task.planning_horizon} steps "
            f"over a {controller.task.planning_horizon * controller.task.dt} second horizon."
        )
    node.get_logger().info("Jitting controller...")

    # break aliasing for mjx_data
    # mjx_data = mjx.forward(controller.task.model, mjx_data)
    # def break_aliasing(tree):
    #     return jax.tree.map(lambda x: x + jnp.zeros_like(x), tree)
    # mjx_data = break_aliasing(mjx_data)
    node.get_logger().info("Broke aliasing for mjx_data...")

    robot_q, robot_dq, T_lin, T_quat = node.get_state_data()
    def step_fn(mjx_data, policy_params, lin_t, quat_t, robot_q, robot_dq):
        # all jnp operations
        lin_t = jnp.asarray(lin_t, jnp.float32)
        quat_t = jnp.asarray(quat_t, jnp.float32)
        robot_q = jnp.asarray(robot_q, jnp.float32)
        robot_dq = jnp.asarray(robot_dq, jnp.float32)

        qpos = mjx_data.qpos.at[0:14].set(jnp.array([
            lin_t[0] - 0.15,
            lin_t[1],
            lin_t[2],
            quat_t[3],
            quat_t[0],
            quat_t[1],
            quat_t[2],
            robot_q[0],
            robot_q[1],
            robot_q[2],
            robot_q[3],
            robot_q[4],
            robot_q[5],
            robot_q[6],
        ], jnp.float32))

        qvel = mjx_data.qvel.at[7:14].set(jnp.array([
            robot_dq[0],
            robot_dq[1],
            robot_dq[2],
            robot_dq[3],
            robot_dq[4],
            robot_dq[5],
            robot_dq[6],
        ], jnp.float32))

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

        policy_params, rollouts = controller.optimize(mjx_data, policy_params)
        return mjx_data, policy_params, rollouts

    jit_step = jax.jit(step_fn, donate_argnums=(0, 1))
    jit_step = jit_step.lower(
        mjx_data,
        policy_params,
        jnp.zeros(3, jnp.float32),  # lin_t shape
        jnp.zeros(4, jnp.float32),  # quat_t
        jnp.zeros_like(robot_q),    # robot_q
        jnp.zeros_like(robot_dq),   # robot_dq
    ).compile()
    st = time.time()
    # jit_optimize = jax.jit(controller.optimize, donate_argnums=(1,))
    # jit_optimize = jit_optimize.lower(mjx_data, policy_params).compile()
    # node.get_logger().info("Warming up JIT...")
    # for _ in range(5):
    #     policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    node.get_logger().info(f"Time to jit: {time.time() - st:.3f} seconds") 


    ###########################
    ##    Planning Loop      ##
    ###########################
    step = 0
    # run controller
    while True:
        step += 1
        # start_time = time.time()
        # # update data
        robot_q, robot_dq, T_lin, T_quat = node.get_state_data()
        # # if any is None
        # if robot_q is None or robot_dq is None or T_lin is None or T_quat is None:
        #     node.get_logger().info("Waiting for state data...")
        #     time.sleep(0.1)
        #     continue

        # mjx_data = set_data(
        #     mjx_data,
        #     T_lin,
        #     T_quat,
        #     robot_q,
        #     robot_dq
        # )

        plan_start = time.time()
        mjx_data, policy_params, rollouts = jit_step(
            mjx_data, policy_params, T_lin, T_quat, robot_q, robot_dq
        )
        # policy_params, rollouts = jit_optimize(mjx_data, policy_params)
        plan_time = time.time() - plan_start
        node.get_logger().info(
            f"Step {step}: Planning time: {plan_time:.3f} seconds"
        )

        u = controller.get_action(policy_params, 0.0)

        node._set_action(np.array(u))



def set_data(mjx_data, lin_t, quat_t, robot_q, robot_dq):

    # (CPU->GPU copy)
    lin_t = jnp.asarray(lin_t, dtype=jnp.float32)
    quat_t = jnp.asarray(quat_t, dtype=jnp.float32)
    robot_q = jnp.asarray(robot_q, dtype=jnp.float32)
    robot_dq = jnp.asarray(robot_dq, dtype=jnp.float32)

    # --- update mjx_data.qpos / qvel on device ---
    new_qpos = mjx_data.qpos.at[0:14].set(jnp.array([
        lin_t[0] - 0.15,
        lin_t[1],
        lin_t[2],
        quat_t[3],
        quat_t[0],
        quat_t[1],
        quat_t[2],
        robot_q[0],
        robot_q[1],
        robot_q[2],
        robot_q[3],
        robot_q[4],
        robot_q[5],
        robot_q[6],
    ], dtype=jnp.float32))

    new_qvel = mjx_data.qvel.at[7:14].set(jnp.array([
        robot_dq[0],
        robot_dq[1],
        robot_dq[2],
        robot_dq[3],
        robot_dq[4],
        robot_dq[5],
        robot_dq[6],
    ], dtype=jnp.float32))

    mjx_data = mjx_data.replace(
        qpos=new_qpos,
        qvel=new_qvel,
    )
    return mjx_data


class PushT_Node(FrankaPandaServer):

    def __init__(self, 
                 robot_ip,
                 ):
        
        super().__init__(robot_ip, 
                         None) 
        
        self.get_logger().info(
            "Initializing SMPC Controller..."
        )
        
        
        # import jax
        # jax.config.update("jax_platform_name", "gpu")
        from mujoco import mjx
        
        ####################################
        ##    MoveIt Safety Constraints   ##    
        ####################################
        self.add_collision_primitive(
            id="table",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(1.2, 0.9, 0.1),
            position=np.array([0.4, 0.0, -0.05]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.add_collision_primitive(
            id="wall x",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(0.05, 1.0, 0.5),
            position=np.array([1.025, 0.0, 0.15]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.add_collision_primitive(
            id="wall y_neg",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(1.2, 0.05, 0.5),
            position=np.array([0.4, -0.475, 0.15]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.add_collision_primitive(
            id="wall y_pos",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(1.2, 0.05, 0.5),
            position=np.array([0.4, 0.475, 0.15]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        ####################################
        ##       Move to initial pose     ##    
        ####################################
        
        self.init_pos = np.array([0.5, 0.1, 0.265])   # 0.165
        # self.init_pos = np.array([0.5, 0.0, 0.255])   # 0,26
        # add small noise: keep x small, increase variance in y
        # self.init_pos[0] += np.random.uniform(-0.1, 0.05)   # x
        # self.init_pos[1] += np.random.uniform(-0.1, 0.1)   # y (larger variance)
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0])
       
        self.init_rot = R.from_quat(self.init_quat).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = self.init_rot
        pose[:3, 3] = self.init_pos
        self.plan_and_move_to_pose(pose)

        # Wait until all states are received
        while not self._states_received():
            rclpy.spin_once(self, timeout_sec=0.1)

        ####################################
        ##            T Object            ##    
        ####################################

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.br = StaticTransformBroadcaster(self)
        self._publish_static_robot_tf()
   

        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tf_buffer.can_transform("fr3_link0", "objectPushT_MuJoCo",
                                            rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=0.0)):
                break
        time.sleep(0.5)  # wait a bit more
        
        ####################################
        ##           Start Timer          ##    
        ####################################

        # 
        self.T_lin = None
        self.T_quat = None
        self.robot_q = None
        self.robot_dq = None

        self.current_action = None

        self.servo_freq = 50  
        self.update_freq = 20
        
        self.sim_group = MutuallyExclusiveCallbackGroup()

        self.create_timer(1.0 / self.update_freq, self._update_state_data, callback_group=self.sim_group)
        self.create_timer(1.0 / self.servo_freq, self._send_command, callback_group=self.parallel_group)
        
    def get_state_data(self):
        return self.robot_q, self.robot_dq, self.T_lin, self.T_quat
    
    def _set_action(self, action):
        """
        Set the action for the servo controller.
        
        Args:
            action (np.ndarray): Action array of shape (2,) representing linear velocities in x and y directions.
        """
        self.current_action = action

    def _publish_static_robot_tf(self):
        """
        Publish static transforms:
            
        1. Transform from 'fr3_link0' (robot base) to 'optitrack' (motion capture system)
            - Based on calibration data

        2. Transform from 'objectPushT' (real object) to 'objectPushT_MuJoCo' (simulated object)
            - Based on difference between centre of obtitrack rigid body and mujoco body 

        Note:
            The transforms are latched by the broadcaster, so they persist after publishing.

        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'fr3_link0'
        t.child_frame_id = 'optitrack'

        t.transform.translation.x = 1.12763
        t.transform.translation.y = -1.26957
        t.transform.translation.z = -0.02129

        t.transform.rotation.x = -0.00703
        t.transform.rotation.y = -0.00123
        t.transform.rotation.z = 0.99989
        t.transform.rotation.w = 0.01335

        self.static_tf = t
        self.br.sendTransform(t)
        self.get_logger().info('Published static TF fr3_link0 -> optitrack')

        # optitrack rigid body to mujoco geom body
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'objectPushT'
        t.child_frame_id = 'objectPushT_MuJoCo'

        t.transform.translation.x = 0.0
        t.transform.translation.y = -0.025
        t.transform.translation.z = 0.0

        quat = quaternion_from_euler(0.0, 0.0, np.pi)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)
        self.get_logger().info('Published static TF objectPushT -> objectPushT_MuJoCo')
    
    def _update_T(self):
        if not self.tf_buffer.can_transform("fr3_link0", "objectPushT_MuJoCo",
                                            rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=0.0)):
            raise RuntimeError("TF transform not available yet.")
        else:
            world_T_objReal = self.tf_buffer.lookup_transform(
                "fr3_link0", "objectPushT_MuJoCo", rclpy.time.Time()
            )
            lin = np.array([
                world_T_objReal.transform.translation.x,
                world_T_objReal.transform.translation.y,
                world_T_objReal.transform.translation.z
            ], dtype=np.float32)

            quat = np.array([
                world_T_objReal.transform.rotation.x,
                world_T_objReal.transform.rotation.y,
                world_T_objReal.transform.rotation.z,
                world_T_objReal.transform.rotation.w
            ], dtype=np.float32)

        return lin, quat

    def _get_robot_state_np(self):
        robot_q = np.array(
            copy.deepcopy(self._current_joint_state.position),
            dtype=np.float32
        )
        robot_dq = np.array(
            copy.deepcopy(self._current_joint_state.velocity),
            dtype=np.float32
        )
        return robot_q, robot_dq
    
    def _update_state_data(self):
        self.robot_q, self.robot_dq = self._get_robot_state_np()
        self.T_lin, self.T_quat = self._update_T()

    def _send_command(self):
        """
        Send velocity command (twist) to moveit servo
        
        :param self: Description
        """
        now_sec = self.get_clock().now().nanoseconds / 1e9
        vx = 0.25*sin(now_sec / 2)
        vy = 0.25*cos(now_sec / 2)
        self.servo(linear=(vx, vy, 0.0), angular=(0.0, 0.0, 0.0))

    def _states_received(self):
        """
        Check if all required states have been received at least once
        """
        if self._current_joint_state is None:
            self.get_logger().warn(
                "Current joint state not received yet."
            )
            return False
        else:
            self.get_logger().info("Current joint state received.")
            return True
    


if __name__ == '__main__':
    
    rclpy.init()
    seed = 42
    
    node = PushT_Node(
        robot_ip="10.90.90.77",
    )

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)

    thread = threading.Thread(target=planning_loop, 
                              args=(seed, node), 
                              daemon=True)
    thread.start()
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down controller...")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
