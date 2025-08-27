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
from geometry_msgs.msg import PoseStamped
from franka_panda_server import FrankaPandaServer

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_pose


class FR3_PushT_SMPC_Controller(FrankaPandaServer):

    def __init__(self, 
                 ctrl: SamplingBasedController,
                 robot_ip,
                 seed
                 ):
        
        super().__init__(robot_ip, 
                         None) 
        
        print("Initializing SMPC Controller...")
        
        # TODO init subscriber for optitrack data
        self._T_mocap = None
        # self._ee_pose_subscriber = self.create_subscription(
        #     PoseStamped,
        #     "/franka_robot_state_broadcaster/current_pose",
        #     self._ee_pose_callback,
        #     10
        # )
        
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
        
        
        ####################################
        ##       Move to initial pose     ##    
        ####################################
        self.move_to_home()
        # wait 
        time.sleep(2.0)
        
        self.init_pos = np.array([0.4, 0.0, 0.65])
        self.init_quat = np.array([1.0, 0.0, 0.0, 0.0])
        # self.init_quat = np.array([0.0, 0.7071, 0.7071, 0.0])  # [x, y, z, w]
        # # quat to rotation matrix
        self.init_rot = R.from_quat(self.init_quat).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = self.init_rot
        pose[:3, 3] = self.init_pos
        self.plan_and_move_to_pose(pose)
        
        time.sleep(2.0)
        
        # TODO safe resulting configuration as home
        
        
        ####################################
        ##            T Object            ##    
        ####################################
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
        
        ####################################
        ##         Set up Timers          ##    
        ####################################

        # SMPC update 
        self.mpc_freq = 20  # Hz
        self.create_timer(1.0 / self.mpc_freq, self._run_controller)

        # # Command publisher
        # self.action_timer = time.time()
        self.get_logger().info("Starting servo...")
        self.servo.enable_servo()
        self.servo.use_twist()  # switch to twist commands
        self.servo_freq = 35  # Hz
        self.create_timer(1.0 / self.servo_freq, self._send_command)
        
        # TODO - regularly check for error threshold and send robot home if below threshold
        
        

        
    def _print(self):
        self._print_joint_states()

    def _update_state(self):
        
        try:
            # Use rclpy.time.Time() for "latest" available transform
            tf = self.tf_buffer.lookup_transform(
                "optitrack", "objectPushT", rclpy.time.Time())
            tr = tf.transform.translation
            q = tf.transform.rotation

            self.latest_translation = (tr.x, tr.y, tr.z)
            self.latest_rotation = (q.x, q.y, q.z, q.w)
            print(f"Object position: {self.latest_translation}, orientation: {self.latest_rotation}")   
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
        
        dq = np.squeeze(np.array([copy.deepcopy(self.mjx_data.qvel)]))
        q = np.squeeze(np.array([copy.deepcopy(self.mjx_data.qpos)]))

        # print end effector pose matrix
        w_H_ee = self.get_ee_pose()
        print(f"End effector pose: \n{w_H_ee}")
        # convert to quaternion
        quat_xyzw = R.from_matrix(w_H_ee[:3, :3]).as_quat()
        print(f"End effector quaternion: {quat_xyzw}")    
                
        with self._lock:
            dq[3:-2] = np.array([copy.deepcopy(self._current_joint_state.velocity)])
            q[3:-2] = np.array([copy.deepcopy(self._current_joint_state.position)])
        
        # TODO update T 
        # ...
        
        self.mjx_data = self.mjx_data.replace(
            qpos=q,
            qvel=dq,
        )
        
    def _send_command(self):
        
        # t = time.time() - self.action_timer
        # action = self.get_action(self.policy_params, t)
        # linear = (action[0], action[1], 0.0)
        # angular = (0.0, 0.0, 0.0)
        # self.servo(linear=linear, angular=angular)
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.servo(linear=(sin(now_sec), cos(now_sec), 0.0), angular=(0.0, 0.0, 0.0))
        

    def _run_controller(self):
        st = time.time()
        # TODO update state
        self._update_state()
        
        # TODO compute action from controller
        self.policy_params = self.jit_optimize(self.mjx_data, self.policy_params)
        u = self.ctrl.get_action(self.policy_params, 0.0)
        
        # TODO send action to robot
        print(f"Action: {u}")
        freq = 1 / (time.time() - st)
        print(f"Controller running at {freq:.3f} Hz")
        self.action_timer = time.time()
        
    
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
    
        
    controller = FR3_PushT_SMPC_Controller(
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
