import os
import time
import copy
from pprint import pformat
import argparse
from math import sin, cos

from hydrax.algs import MPPI, MTP, AnMTP
from pusht_franka_free import PushTFranka

from hydrax.alg_base import SamplingBasedController

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
import jax.numpy as jnp

from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Point
from franka_panda_server import FrankaPandaServer

from rclpy.callback_groups import ReentrantCallbackGroup
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix
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
                 debug_model: None,
                 debug_data: None,
                 viewer: None
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
        
        self.init_pos = np.array([0.5, 0.1, 0.158])   # 0,26
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
        
        ####################################
        ##         Debug Simulator        ##    
        ####################################
        self.debug_model = debug_model
        self.debug_data = debug_data
        self.viewer = viewer
        self.servo_freq = 50  # Hz
        self.sim_freq = 20
        self.servo_group = ReentrantCallbackGroup()
        self.sim_group = ReentrantCallbackGroup()
        time.sleep(0.1) 
        self.create_timer(1.0 / self.sim_freq, self._step_debug_sim, callback_group=self.sim_group)
        time.sleep(0.1) 
        self.create_timer(1.0 / self.servo_freq, self._send_command, callback_group=self.servo_group)
        

    def _step_debug_sim(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        self._update_state()
        if not self.viewer.is_running():
            self.get_logger().info("Viewer closed — shutting down.")
            # Cancel timer first to avoid callbacks during shutdown.
            rclpy.shutdown()
            return

        # Step simulation then sync the viewer.
     
        mujoco.mj_forward(self.debug_model, self.debug_data)
        self.viewer.sync()
        current_time2 = self.get_clock().now().nanoseconds / 1e9
        freq = 1 / (current_time2 - current_time)
        print(f"Debug sim running at {freq:.3f} Hz")
  
    
    
    def _publish_static_robot_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'fr3_link0'
        t.child_frame_id = 'optitrack'

        # Translation (meters)
        # t.transform.translation.x = 1.12763
        # t.transform.translation.y = -1.26957
        # t.transform.translation.z = -0.02129
        t.transform.translation.x = 1.07658
        t.transform.translation.y = -1.23784
        t.transform.translation.z = 0.04381

        # t.transform.rotation.x = -0.00703
        # t.transform.rotation.y = -0.00123
        # t.transform.rotation.z = 0.99989
        # t.transform.rotation.w = 0.01335
        t.transform.rotation.x = -0.01901
        t.transform.rotation.y = 0.00215
        t.transform.rotation.z = 0.99975
        t.transform.rotation.w = -0.01119

        # Broadcast once; static transforms are latched
        self.static_tf = t
        self.br.sendTransform(t)
        self.get_logger().info('Published static TF fr3_link0 -> optitrack')

        # same for mujoco
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'objectPushT'
        t.child_frame_id = 'objectPushT_MuJoCo'
        # objectPushT, fr3-calibration-ee
        
        # TODO hardcoded for now, potentially automatically publish after calibration in the future

        # Translation (meters)
        t.transform.translation.x = 0.0
        t.transform.translation.y = +0.025
        t.transform.translation.z = 0.0

        quat = quaternion_from_euler(0.0, 0.0, np.pi)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        # Broadcast once; static transforms are latched
        self.br.sendTransform(t)
        self.get_logger().info('Published static TF objectPushT -> objectPushT_MuJoCo')
    
        
    def _update_T(self):
        if not self.tf_buffer.can_transform("fr3_link0", "objectPushT_MuJoCo",
                                            rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=0.1)):
            raise RuntimeError("TF transform not available yet.")
        else:
            world_T_objReal = self.tf_buffer.lookup_transform("fr3_link0", "objectPushT_MuJoCo", rclpy.time.Time())
            lin = np.array([world_T_objReal.transform.translation.x,
                            world_T_objReal.transform.translation.y,
                            world_T_objReal.transform.translation.z])   
             
            quat = np.array([world_T_objReal.transform.rotation.x,
                             world_T_objReal.transform.rotation.y,
                             world_T_objReal.transform.rotation.z,
                             world_T_objReal.transform.rotation.w])
    
        return lin, quat


    def _update_state(self):
        
        lin_t, quat_t = self._update_T()

        self.debug_data.qpos[[0, 1, 3, 4, 5, 6]] = [lin_t[0],
                                    lin_t[1], 
                                    quat_t[3], 
                                    quat_t[0], 
                                    quat_t[1],
                                    quat_t[2]]

        # robot jints are in qpos[1:7]
        self.debug_data.qpos[7:14] = np.array([copy.deepcopy(self._current_joint_state.position)])
        self.debug_data.qvel[7:14] = np.array([copy.deepcopy(self._current_joint_state.velocity)])

        # TODO - twist based on frame estimation history

        
    def _send_command(self):

        # self.servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
        now_sec = self.get_clock().now().nanoseconds / 1e9
        vx = 0.15*sin(now_sec / 2)
        vy = 0.15*cos(now_sec / 2)
        self.servo(linear=(vx, vy, 0.0), angular=(0.0, 0.0, 0.0))
        

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
    
    



if __name__ == '__main__':
    
    rclpy.init()

    task = PushTFranka(ik_type = 'pinv',
                    planning_horizon=12,
                    sim_steps_per_control_step=2,
                    ctrl_limits={"u_min": jnp.array([-0.4, -0.4]), 
                                 "u_max": jnp.array([0.4, 0.4])},
                    actuation_type='velocity',
                    sampling_space="velocity",
                    block_type = 'sim-real',
                )

    import mujoco
    import mujoco.viewer
    
 
    # Load the MuJoCo model
    xml_path = "/home/mtp/Lab/mtp-thesis/hydrax/models/fr3_pushT_vel/scene_mjx_free.xml"
    # xml_path = "/home/magnus/GitHub/mtp/hydrax/hydrax/models/pusht_franka/scene.xml"
    xml_dir = os.path.dirname(xml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
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
    
    with mujoco.viewer.launch_passive(model, data) as v:
        controller = FR3_PushT(
            ctrl=ctrl,
            robot_ip="10.90.90.77",
            seed=seed,
            debug_model=model,
            debug_data=data,
            viewer=v,
        )

        # TODO - does multi-threaded executor give me any advantage ? -> Benchmark
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=8)
        executor.add_node(controller)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            print("Shutting down controller...")
        finally:
            executor.shutdown()
            controller.destroy_node()
            rclpy.shutdown()
