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

import math

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
        print("Backend:", jax.default_backend())

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
            lambda d, p: ctrl.optimize(d, p)[0], donate_argnums=(0,1)
        )
        self.get_action = jax.jit(ctrl.get_action)
        self.policy_params = self.jit_optimize(self.mjx_data, self.policy_params)
        print(f"Time to jit: {time.time() - st}")


        ####################################
        ##         Create Buffers         ##    
        ####################################

        nq = self.mjx_data.qpos.shape[0]
        nv = self.mjx_data.qvel.shape[0]
        self._q_buf  = np.empty(nq, dtype=self.mjx_data.qpos.dtype)
        self._dq_buf = np.empty(nv, dtype=self.mjx_data.qvel.dtype)
        self._tmp_lin  = np.empty(3, dtype=np.float64)
        self._tmp_quat = np.empty(4, dtype=np.float64)

        # cache constants / indices
        self._idx_robot_start = 3
        self._idx_robot_end   = -2

        # Optional: set a flag to silence loop logs
        self._log_every = 20
        self._tick = 0
        
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

        
    def _get_T_pose(self):
        if not self.tf_buffer.can_transform("fr3_link0", "objectPushT", rclpy.time.Time(), rclpy.duration.Duration(seconds=0.01)):
            return None  # signal "no update"
        t = self.tf_buffer.lookup_transform("fr3_link0", "objectPushT", rclpy.time.Time())
        tr = t.transform.translation
        rq = t.transform.rotation
        # optitrack vs mujoco center offset
        self._tmp_lin[0] = tr.x
        self._tmp_lin[1] = tr.y - 0.025
        self._tmp_lin[2] = tr.z
        self._tmp_quat[0] = rq.x
        self._tmp_quat[1] = rq.y
        self._tmp_quat[2] = rq.z
        self._tmp_quat[3] = rq.w
        return self._tmp_lin, self._tmp_quat


    def _update_state(self):
        # Fill buffers in place (no new arrays)
        # 1) Object T
        tvals = self._get_T_pose()
        if tvals is None:
            return False
        lin_t, quat_t = tvals
        self._q_buf[:] = self.mjx_data.qpos
        self._dq_buf[:] = self.mjx_data.qvel

        # map to model
        self._q_buf[0] = -lin_t[1]
        self._q_buf[1] =  lin_t[0] - 0.5
        self._q_buf[2] = math.atan2(2*(quat_t[3]*quat_t[2] + quat_t[0]*quat_t[1]),
                                    1 - 2*(quat_t[1]*quat_t[1] + quat_t[2]*quat_t[2])) - math.pi

        # 2) Robot joints
        with self._lock:
            js = self._current_joint_state
            if js is None:
                return False
            # js.position, js.velocity are sequences
            s, e = self._idx_robot_start, self._idx_robot_end
            self._q_buf[s:e]  = js.position
            self._dq_buf[s:e] = js.velocity

        # 3) Replace mjx data once (no new shapes)
        self.mjx_data = self.mjx_data.replace(qpos=self._q_buf, qvel=self._dq_buf)
        return True
        
        
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

        if not self._update_state():
            return

        # JITed path
        self.policy_params = self.jit_optimize(self.mjx_data, self.policy_params)
        self._current_u = self.get_action(self.policy_params, 0.0)

        self._tick += 1
        if (self._tick % self._log_every) == 0:
            dt = time.time() - st
            self.get_logger().info(f"SMPC loop: {1.0/dt:.1f} Hz, dt={dt*1e3:.2f} ms")
    
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
