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

import jax
jax.config.update("jax_platform_name", "gpu")
import jax.numpy as jnp

from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Point
from franka_panda_server import FrankaPandaServer

from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import quaternion_from_euler, quaternion_multiply, quaternion_matrix
from tf2_geometry_msgs import do_transform_pose
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster





class FR3_PushT(FrankaPandaServer):

    def __init__(self, 
                 ctrl: SamplingBasedController,
                 robot_ip,
                 seed,
                 ):
        
        super().__init__(robot_ip, 
                         None) 
        
        print("Initializing SMPC Controller...")
        
        
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
        
        self.init_pos = np.array([0.45, 0.1, 0.165])   # 0,26
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
        ##         JIT Controller         ##    
        ####################################
        # Wait until all states are received
        while not self._states_received():
            rclpy.spin_once(self, timeout_sec=0.1)

        print("All states received, initializing controller...")

        self.ctrl = ctrl

        # Create mjx_data once on host, then push to device via jitted code
        self.mjx_data = mjx.make_data(self.ctrl.task.model)
        self.policy_params = self.ctrl.init_params(seed)

        print(
            f"Planning with {self.ctrl.task.planning_horizon} steps "
            f"over a {self.ctrl.task.planning_horizon * self.ctrl.task.dt} second horizon."
        )

        print("Jitting controller...")
        st = time.time()

        # Do a forward once (host side is fine here)
        self.mjx_data = mjx.forward(ctrl.task.model, self.mjx_data)

        # Make unified jitted step function
        self.jit_step = self.make_jitted_step(ctrl)

        # Warmstart on device with dummy data (zeros)
        lin0 = jnp.zeros(3, dtype=jnp.float32)
        quat0 = jnp.array([0., 0., 0., 1.], dtype=jnp.float32)
        q0 = jnp.zeros(7, dtype=jnp.float32)
        dq0 = jnp.zeros(7, dtype=jnp.float32)

        # One call to transfer mjx_data & policy_params to GPU and compile
        self.mjx_data, self.policy_params = self.jit_step(
            self.mjx_data, self.policy_params,
            lin0, quat0, q0, dq0
        )

        # Extra warmstart iterations if you want
        for _ in range(4):
            self.mjx_data, self.policy_params = self.jit_step(
                self.mjx_data, self.policy_params,
                lin0, quat0, q0, dq0
            )

        import jax
        jax.block_until_ready(self.policy_params)
        self.device = jax.devices("gpu")[0]
        print(f"Controller jit and warmstarted on device: {self.device}")
        print(f"Time to jit and warmstart: {time.time() - st:.3f} s")


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

        self.servo_freq = 50  # Hz
        self.plan_freq = 10
        
        self.servo_group = ReentrantCallbackGroup()
        self.sim_group = MutuallyExclusiveCallbackGroup()

        self.create_timer(1.0 / self.plan_freq, self._run_controller, callback_group=self.sim_group)
        self.create_timer(1.0 / self.servo_freq, self._send_command, callback_group=self.servo_group)
        
    
    def _publish_static_robot_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'fr3_link0'
        t.child_frame_id = 'optitrack'

        # Translation (meters)
        t.transform.translation.x = 1.12763
        t.transform.translation.y = -1.26957
        t.transform.translation.z = -0.02129

        t.transform.rotation.x = -0.00703
        t.transform.rotation.y = -0.00123
        t.transform.rotation.z = 0.99989
        t.transform.rotation.w = 0.01335

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
        t.transform.translation.y = -0.025
        t.transform.translation.z = 0.0

        quat = quaternion_from_euler(0.0, 0.0, np.pi)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        # Broadcast once; static transforms are latched
        self.br.sendTransform(t)
        self.get_logger().info('Published static TF objectPushT -> objectPushT_MuJoCo')
    
    def make_jitted_step(self, ctrl):

        def step(mjx_data, policy_params,
                lin_t, quat_t,
                robot_q, robot_dq):
            """
            mjx_data, policy_params: persistent state (kept on device)
            lin_t: (3,) position (np or jnp)
            quat_t: (4,) quaternion (x,y,z,w)
            robot_q, robot_dq: (7,) joint pos/vel
            """

            # Convert small inputs to jax arrays (cheap CPU->GPU copy)
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

            # --- run the controller on device ---
            new_policy_params, *rest = ctrl.optimize(mjx_data, policy_params)
            # (assuming optimize returns (params, other_stuff...))

            return mjx_data, new_policy_params

        # Donate both mjx_data (arg 0) and policy_params (arg 1)
        return jax.jit(step, donate_argnums=(1,))


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
        # small helper
        robot_q = np.array(
            copy.deepcopy(self._current_joint_state.position),
            dtype=np.float32
        )
        robot_dq = np.array(
            copy.deepcopy(self._current_joint_state.velocity),
            dtype=np.float32
        )
        return robot_q, robot_dq

    def _run_controller(self):
        import jax

        t0 = time.time()
        lin_t, quat_t = self._update_T()
        robot_q, robot_dq = self._get_robot_state_np()
        t1 = time.time()

        self.mjx_data, self.policy_params = self.jit_step(
            self.mjx_data,
            self.policy_params,
            lin_t,
            quat_t,
            robot_q,
            robot_dq,
        )
        jax.block_until_ready(self.policy_params)
        t2 = time.time()

        print(
            f"obs (TF + joint state): {(t1 - t0)*1000:.1f} ms, "
            f"jit_step (GPU + JAX): {(t2 - t1)*1000:.1f} ms, "
            f"total: {(t2 - t0)*1000:.1f} ms"
        )
    def _update_state(self):
        
        lin_t, quat_t = self._update_T()

        # TODO velocity estimation from history of frames
        robot_q = jnp.array(copy.deepcopy(self._current_joint_state.position))
        robot_dq = jnp.array(copy.deepcopy(self._current_joint_state.velocity))

        new_q = self.mjx_data.qpos.at[0:14].set(jnp.array([
                lin_t[0] - 0.15,  #! Why ?
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
            ]))
        new_dq = self.mjx_data.qvel.at[7:14].set(jnp.array([
                robot_dq[0],
                robot_dq[1],
                robot_dq[2],
                robot_dq[3],
                robot_dq[4],
                robot_dq[5],
                robot_dq[6],
            ]))

        self.mjx_data = self.mjx_data.replace(
            qpos=new_q,
            qvel=new_dq,
        )

        
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
                    block_type = 'free',
                )

    import mujoco
    
 
    # Load the MuJoCo model
    xml_path = "/home/mtp/Lab/mtp-thesis/hydrax/models/fr3_pushT_vel/scene_mjx_free.xml"
    xml_dir = os.path.dirname(xml_path)
    
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
            num_samples=512,
            noise_level=0.3,
            temperature=0.1,
            num_randomizations=2,
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
        robot_ip="10.90.90.77",
        seed=seed,
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
