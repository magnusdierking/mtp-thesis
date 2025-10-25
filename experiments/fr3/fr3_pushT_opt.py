import math, time, argparse, numpy as np
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
os.environ.setdefault("JAX_ENABLE_X64", "0")  # prefer fp32
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from shape_msgs.msg import SolidPrimitive

from hydrax.algs import MPPI, MTP
from hydrax.tasks.pusht_franka import PushTFranka
from hydrax.alg_base import SamplingBasedController

from franka_panda_server import FrankaPandaServer

import jax
import jax.numpy as jnp

from mujoco import mjx
import concurrent.futures

# ---------- Helpers ----------
def yaw_from_quat(x, y, z, w):
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


# ============= Node =============
class FR3_PushT(FrankaPandaServer):
    def __init__(self, ctrl: SamplingBasedController, robot_ip: str, seed: int):
        super().__init__(robot_ip, None)

        # --- Safety primitives (MoveIt) ---
        self.add_collision_primitive(
            id="table",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(2.0, 0.8, 0.1),
            position=np.array([0.0, 0.0, -0.05]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.add_collision_primitive(
            id="wall x",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(0.1, 0.8, 0.4),
            position=np.array([1.05, 0.0, 0.1]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.add_collision_primitive(
            id="wall y_neg",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(2.0, 0.1, 0.4),
            position=np.array([0.0, -0.45, 0.1]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.add_collision_primitive(
            id="wall y_pos",
            primitive_type=SolidPrimitive.BOX,
            dimensions=(2.0, 0.1, 0.4),
            position=np.array([0.0, 0.45, 0.1]),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )

        # --- TF setup ---
        self.br = StaticTransformBroadcaster(self)
        self._publish_static_robot_tf()
        time.sleep(0.3)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Go to initial pose (blocking call outside timers) ---
        init_pos = np.array([0.4, 0.0, 0.28])
        init_pos[0] += np.random.normal(0, 0.01)
        init_pos[1] += np.random.normal(0, 0.05)
        init_quat = np.array([1.0, 0.0, 0.0, 0.0])
        pose = np.eye(4); pose[:3,:3] = R.from_quat(init_quat).as_matrix(); pose[:3,3] = init_pos
        self.plan_and_move_to_pose(pose)
        time.sleep(1.0)

        # Wait for first joint state (don’t block timers later)
        while not self._states_received():
            rclpy.spin_once(self, timeout_sec=0.1)

        # --- Controller / JAX ---
        self.ctrl = ctrl
        model_f32 = mjx.convert_model(ctrl.task.model, dtype=jnp.float32)
        self.mjx_data = mjx.forward(model_f32, mjx.make_data(model_f32))
        ctrl.task.model = model_f32  # keep everything consistent
        # self.mjx_data = mjx.forward(ctrl.task.model, mjx.make_data(ctrl.task.model))
        self.policy_params = ctrl.init_params(seed)
        self.get_logger().info(
            f"Backend: {jax.default_backend()} | Horizon {ctrl.task.planning_horizon} "
            f"({ctrl.task.planning_horizon * ctrl.task.dt:.3f}s)"
        )

        # Precompile optimize -> returns new params (donate only params)
        self.get_logger().info("Jitting controller (optimize + get_action)... this may take a while.")
        t0 = time.time()
        # self._jit_optimize = jax.jit(lambda d, p: ctrl.optimize(d, p)[0], donate_argnums=(1,))
        # self._executable = self._jit_optimize.lower(self.mjx_data, self.policy_params).compile()
        self._jit_optimize = jax.jit(
                                lambda d, p: ctrl.optimize(d, p)[0],
                                donate_argnums=(1,), # doante both
                            )
        self._executable = self._jit_optimize.lower(self.mjx_data, self.policy_params).compile()
        self._get_action = ctrl.get_action
        t1 = time.time()
        self.get_logger().info(f"JIT compile finished in {t1 - t0:.2f} s")


        # Buffers (numpy, reused)
        nq, nv = self.mjx_data.qpos.shape[0], self.mjx_data.qvel.shape[0]
        self._q_buf = np.empty(nq, dtype=self.mjx_data.qpos.dtype)
        self._dq_buf = np.empty(nv, dtype=self.mjx_data.qvel.dtype)
        self._tmp_lin = np.empty(3, dtype=np.float64)
        self._tmp_quat = np.empty(4, dtype=np.float64)
        self._idx_robot_start, self._idx_robot_end = 3, -2
        # device buffer
        self._q_dev  = jnp.array(self.mjx_data.qpos)  # on device
        self._dq_dev = jnp.array(self.mjx_data.qvel)

        # Command state
        self._current_u = np.zeros((2,), dtype=np.float32)
        self._tick = 0

        # Logging
        self._servo_tick = 0
        self._planner_tick = 0
        self._log_every = 10   # log every N iterations
        self._last_servo_time = time.time()
        self._last_plan_time = time.time()
    

        # --- Timers & threading ---
        self.mpc_freq = 3.0     # Hz (planner)
        self.servo_freq = 50.0  # Hz (publisher)

        self.servo_group   = ReentrantCallbackGroup()
        self.planner_group = MutuallyExclusiveCallbackGroup()

        # Servo setup
        self.get_logger().info("Starting servo...")
        self.servo.enable_servo()
        self.servo.use_twist()

        # Servo timer: never blocks
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        # (your servo publisher likely uses QoS internally; here we just ensure timer is separate)
        self.create_timer(1.0/self.servo_freq, self._send_command, callback_group=self.servo_group)

        # Planner trigger: submits work to thread pool (non-blocking)
        self.create_timer(1.0/self.mpc_freq, self._plan_trigger, callback_group=self.planner_group)
        self._plan_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._planning = False

    # ---------- Static TF ----------
    def _publish_static_robot_tf(self):
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

        self.br.sendTransform(t)
        self.get_logger().info('Published static TF fr3_link0 -> optitrack')

    # ---------- State update ----------
    def _get_T_pose(self):
        if not self.tf_buffer.can_transform("fr3_link0", "objectPushT",
                                            rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=0.01)):
            return None
        t = self.tf_buffer.lookup_transform("fr3_link0", "objectPushT", rclpy.time.Time())
        tr, rq = t.transform.translation, t.transform.rotation
        self._tmp_lin[:]  = (tr.x, tr.y - 0.025, tr.z)  # small offset
        self._tmp_quat[:] = (rq.x, rq.y, rq.z, rq.w)
        return self._tmp_lin, self._tmp_quat

    # def _update_state(self):
    #     tvals = self._get_T_pose()
    #     if tvals is None:
    #         return False
    #     lin_t, quat_t = tvals
    #     self._q_buf[:] = self.mjx_data.qpos
    #     self._dq_buf[:] = self.mjx_data.qvel
    #     # map object pose
    #     self._q_buf[0] = -lin_t[1]
    #     self._q_buf[1] =  lin_t[0] - 0.44
    #     self._q_buf[2] = math.atan2(2*(quat_t[3]*quat_t[2] + quat_t[0]*quat_t[1]),
    #                                 1 - 2*(quat_t[1]*quat_t[1] + quat_t[2]*quat_t[2])) - math.pi
    #     # robot joints
    #     with self._lock:
    #         js = self._current_joint_state
    #         if js is None:
    #             return False
    #         s, e = self._idx_robot_start, self._idx_robot_end
    #         self._q_buf[s:e]  = js.position
    #         self._dq_buf[s:e] = js.velocity
    #     # update mjx data (host arrays are copied to device inside executable)
    #     self.mjx_data = self.mjx_data.replace(qpos=self._q_buf, qvel=self._dq_buf)
    #     return True
    def _update_state(self):
        tvals = self._get_T_pose()
        if tvals is None:
            return False
        lin_t, quat_t = tvals

        ang = math.atan2(2*(quat_t[3]*quat_t[2] + quat_t[0]*quat_t[1]),
                        1 - 2*(quat_t[1]*quat_t[1] + quat_t[2]*quat_t[2])) - math.pi

        q  = self._q_dev
        dq = self._dq_dev

        # scalar field updates
        q  = q.at[0].set(-lin_t[1])
        q  = q.at[1].set(lin_t[0] - 0.44)
        q  = q.at[2].set(ang)

        with self._lock:
            js = self._current_joint_state
            if js is None:
                return False
            s, e = self._idx_robot_start, self._idx_robot_end
            q  = q.at[s:e].set(jnp.asarray(js.position, dtype=q.dtype))
            dq = dq.at[s:e].set(jnp.asarray(js.velocity, dtype=dq.dtype))

        self._q_dev, self._dq_dev = q, dq
        self.mjx_data = self.mjx_data.replace(qpos=q, qvel=dq)
        return True
    

    # ---------- Planner (non-blocking) ----------
    def _plan_trigger(self):
        if self._planning:
            return
        if not self._update_state():
            return
        self._planning = True
        fut = self._plan_pool.submit(self._do_plan, self.mjx_data, self.policy_params)
        fut.add_done_callback(self._on_plan_done)

    def _do_plan(self, mjx_data, params):
        # Compute next params (compiled, no ROS calls)
        new_params = self._executable(mjx_data, params)
        # Compute action (cheap)
        action = self._get_action(new_params, 0.0)
        return new_params, np.asarray(action, dtype=np.float32)

    def _on_plan_done(self, fut):
        try:
            self.policy_params, action_np = fut.result()
            self._current_u = action_np
            self._planner_tick += 1
            if self._planner_tick % self._log_every == 0:
                now = time.time()
                dt = now - self._last_plan_time
                freq = self._log_every / dt
                self.get_logger().info(f"[Planner] {freq:.1f} Hz (avg over {self._log_every} iters)")
                self._last_plan_time = now
        except Exception as e:
            self.get_logger().error(f"Planner error: {e}")
        finally:
            self._planning = False

    # ---------- Servo (runs regardless of planner) ----------
    def _send_command(self):
        u = self._current_u
        if u is None:
            self.servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
        else:
            self.servo(linear=(float(u[0]), float(u[1]), 0.0), angular=(0.0, 0.0, 0.0))

        # --- Logging ---
        self._servo_tick += 1
        if self._servo_tick % self._log_every == 0:
            now = time.time()
            dt = now - self._last_servo_time
            freq = self._log_every / dt
            self.get_logger().info(f"[Servo]   {freq:.1f} Hz (avg over {self._log_every} iters)")
            self._last_servo_time = now

    # ---------- Boot helpers ----------
    def _states_received(self):
        if self._current_joint_state is None:
            print("Current joint state not received yet.")
            return False
        print("Current joint state received.")
        return True


# ============= Main =============
if __name__ == '__main__':
    rclpy.init()

    # Args
    parser = argparse.ArgumentParser(description="FR3 Push-T SMPC")
    sp = parser.add_subparsers(dest="algorithm")
    sp.add_parser("mppi")
    sp.add_parser("mtp")
    args = parser.parse_args()
    if args.algorithm is None:
        args.algorithm = "mtp"

    # Task + controller
    task = PushTFranka(ik_type = 'pinv',
                       planning_horizon=12,
                       sim_steps_per_control_step=4,
                       ctrl_limits={"u_min": jnp.array([-0.35, -0.35]), 
                                    "u_max": jnp.array([0.35, 0.35])},
                       trace_sites=[],
                       actuation_type='velocity',)
    seed = 42
    if args.algorithm == "mppi":
        ctrl = MPPI(task, 
                    num_samples=256, 
                    noise_level=0.3, 
                    temperature=0.1,
                    num_randomizations=5, 
                    seed=seed)
    else:
        ctrl = MTP(task, 
                   num_samples=64, 
                   M=2, 
                   N=16, 
                   num_elites=12,
                   beta=0.25, 
                   alpha=0.01, 
                   interpolation='bspline',
                   num_randomizations=5, 
                   seed=seed)
        
    print(
        f"Planning with {ctrl.task.planning_horizon} steps "
        f"over a {ctrl.task.planning_horizon * ctrl.task.dt} "
        f"second horizon."
    )

    node = FR3_PushT(ctrl=ctrl, robot_ip="10.90.90.77", seed=seed)

    # Run with two threads so servo and planner can overlap
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
