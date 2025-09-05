import os
import time
import copy
from pprint import pformat
import argparse
from math import sin, cos

from hydrax.algs import MPPI, MTP, AnMTP
from hydrax.tasks.pusht_franka import PushTFranka

from hydrax.alg_base import SamplingBasedController

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R

from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from geometry_msgs.msg import PoseStamped, TransformStamped
from franka_panda_server import FrankaPandaServer

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def yaw_from_quat(x, y, z, w):
    # standard ZYX Euler convention
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw



class FR3_PushT(FrankaPandaServer):

    def __init__(self, 
                 ctrl: SamplingBasedController,
                 robot_ip,
                 seed,
                 ):
        
        super().__init__(robot_ip, 
                         None) 
        
        print("Initializing SMPC Controller...")
        
        
        import jax
        from mujoco import mjx
        
        ####################################
        ##    MoveIt Safety Constraints   ##    
        ####################################
        self.add_collision_primitive(
            id="table",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(1.5, 1, 0.1),
            position=np.array([0.0, 0.0, -0.05]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        # TODO add walls around action space
        
        ####################################
        ##       Move to initial pose     ##    
        ####################################
        # self.move_to_home()
        # # wait 
        # time.sleep(2.0)
        
        self.init_pos = np.array([0.4, 0.0, 0.26])
        # add small noise: keep x small, increase variance in y
        self.init_pos[0] += np.random.normal(0, 0.01)   # x
        self.init_pos[1] += np.random.normal(0, 0.05)   # y (larger variance)
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0])
       
        self.init_rot = R.from_quat(self.init_quat).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = self.init_rot
        pose[:3, 3] = self.init_pos
        self.plan_and_move_to_pose(pose)
        
        time.sleep(2.0)
        
        ####################################
        ##            T Object            ##    
        ####################################
        
        self.br = StaticTransformBroadcaster(self)
        self._publish_static_robot_tf()
        time.sleep(1.0)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        ####################################
        ##         JIT Controller         ##    
        ####################################
        
        # Wait until all states are received
        while not self._states_received():
            rclpy.spin_once(self, timeout_sec=0.1)

        print("All states received, initializing controller...")

        self.ctrl = ctrl
        self.mjx_data = mjx.make_data(self.ctrl.task.model)
        self.policy_params = self.ctrl.init_params(seed)
        print(
            f"Planning with {self.ctrl.task.planning_horizon} steps "
            f"over a {self.ctrl.task.planning_horizon * self.ctrl.task.dt} second horizon."
        )
        print("Jitting controller...")
        print("This may take a while, please be patient.")
        st = time.time()
        self.mjx_data = mjx.forward(ctrl.task.model, self.mjx_data)
        self.jit_optimize = jax.jit(
            lambda d, p: ctrl.optimize(d, p)[0], donate_argnums=(1,)
        )
        self.get_action = jax.jit(ctrl.get_action)
        self.policy_params = self.jit_optimize(self.mjx_data, self.policy_params)
        print(f"Time to jit: {time.time() - st}")
        
        self._current_u = None  # most recent action

        ####################################
        ##         Set up Timers          ##    
        ####################################

        # # SMPC update 
        self.mpc_freq = 20  # Hz
        self.create_timer(1.0 / self.mpc_freq, self._run_controller)

        # # # Command publisher
        # # self.action_timer = time.time()
        self.get_logger().info("Starting servo...")
        self.servo.enable_servo()
        self.servo.use_twist()  # switch to twist commands
        #self.servo_freq = 35  # Hz
        #self.create_timer(1.0 / self.servo_freq, self._send_command)
        
        # # TODO - regularly check for error threshold and send robot home if below threshold

    
    
    def _publish_static_robot_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'fr3_link0'
        t.child_frame_id = 'optitrack'
        
        # TODO hardcoded for now, potentially automatically publish after calibration in the future

        # Translation (meters)
        t.transform.translation.x = 2.59938
        t.transform.translation.y = -0.99226
        t.transform.translation.z = -0.05083

        t.transform.rotation.x = 0.00583
        t.transform.rotation.y = -0.00801
        t.transform.rotation.z = 0.99976
        t.transform.rotation.w = 0.01940

        # Broadcast once; static transforms are latched
        self.static_tf = t
        self.br.sendTransform(t)
        self.get_logger().info('Published static TF fr3_link0 -> optitrack')

        
    def _update_T(self):
        try:
            world_T_objReal = self.tf_buffer.lookup_transform(
                "fr3_link0", "objectPushT", rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            
        # optitrack gives center of markers, need to convert to simulation center
        lin = world_T_objReal.transform.translation
        lin = np.array([lin.x, lin.y - 0.025, lin.z]) # offset due to optitrack vs mujoco center missmatch
        quat = np.array([world_T_objReal.transform.rotation.x,
                         world_T_objReal.transform.rotation.y,
                         world_T_objReal.transform.rotation.z,
                         world_T_objReal.transform.rotation.w])
        # print(f"World transform (translation): {lin}")
    
        return lin, quat


    def _update_state(self):
        
        new_q = np.zeros_like(self.mjx_data.qpos)
        new_dq = np.zeros_like(self.mjx_data.qvel)
        
        # Update T position and orientation 
        lin_t, quat_t = self._update_T()
        new_q[0] = - lin_t[1]     # x in block, -y in robot
        new_q[1] = lin_t[0] - 0.5 # y in block, x in robot, offset from spawn
        new_q[2] = yaw_from_quat(x=quat_t[0], y=quat_t[1], z=quat_t[2], w=quat_t[3]) - np.pi 
        
        # TODO - estimate velocity of T
        
        # Update robot state
        with self._lock:
            if self._current_joint_state is not None:
                new_dq[3:-2] = np.array([copy.deepcopy(self._current_joint_state.velocity)])
                new_q[3:-2] = np.array([copy.deepcopy(self._current_joint_state.position)])
            else:
                print("Warning: Current joint state is None, skipping update.")
                return
        
        self.mjx_data = self.mjx_data.replace(
            qpos=new_q,
            qvel=new_dq,
        )
        
        
    def _send_command(self):
        action = self._current_u
        if action is not None:
            print(f"Sending action: {action}")
            # TODO - scale action 
            linear = (float(action[0]), float(action[1]), 0.0)
            angular = (0.0, 0.0, 0.0)
            self.servo(linear=linear, angular=angular)
        else:
            self.servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))

    def _run_controller(self):
        st = time.time()
        # Update state
        self._update_state()
        
        # Compute action from controller
        self.policy_params = self.jit_optimize(self.mjx_data, self.policy_params)
        self._current_u = self.ctrl.get_action(self.policy_params, 0.0)
        
        # TODO send action to robot
        print(f"Action: {self._current_u}")
        freq = 1 / (time.time() - st)
        print(f"Controller running at {freq:.3f} Hz")
        self.action_timer = time.time()
        
        self._send_command()
        
    
    def _states_received(self):
        # Check if all states have been received
        if self._current_joint_state is None:
            print("Current joint state not received yet.")
            return False
        else:
            print("Current joint state received.")
            return True
    
    
    
    # self.get_logger().debug("This is a debug message")
    #     self.get_logger().info("This is an info message")
    #     self.get_logger().warn("This is a warning")
    #     self.get_logger().error("This is an error")
    #     self.get_logger().fatal("This is fatal!")
    
    
if __name__ == '__main__':
    
    rclpy.init()

    task = PushTFranka(
    )
    
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

    seed = 42

    # Set the controller based on command-line arguments
    if args.algorithm is None: 
        args.algorithm = "mtp"  # Default to MTP
    elif args.algorithm == "mppi":
        print("Running MPPI")
        ctrl = MPPI(
            task,
            num_samples=128,
            noise_level=0.3,
            temperature=0.1,
            num_randomizations=4,
            seed=seed,
        )
    elif args.algorithm == "mtp":
        print("Running MTP")
        ctrl = MTP(
                task,
                num_samples=128,
                M=2, # horizon via control points
                N=64, # samples 
                num_elites=4,
                beta=0.05,
                alpha=0.01,
                interpolation='bspline',
                num_randomizations=2,
                seed=seed,
            )
    
    controller = FR3_PushT(
        ctrl=ctrl,
        robot_ip="10.90.90.144",
        seed=seed,
    )

    # TODO - does multi-threaded executor give me any advantage ? -> Benchmark
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down controller...")
    finally:
        executor.shutdown()
        controller.destroy_node()
        rclpy.shutdown()
