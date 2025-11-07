
import math, time, argparse, os, threading
import numpy as np

# ---- Force GPU if available (JAX 0.4.x uses JAX_PLATFORMS) ----
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")  # compat with older JAX
os.environ.setdefault("JAX_ENABLE_X64", "0")
# Reduce aggressive prealloc to play nice with ROS/GPU sharing
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# Optional: choose a specific GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from shape_msgs.msg import SolidPrimitive

import jax
import jax.numpy as jnp
from jax import device_put

# Use hydrax.mjx (your project wrapper) NOT mujoco.mjx directly

from hydrax.algs import MPPI, MTP
from hydrax.tasks.pusht_franka import PushTFranka
from hydrax.alg_base import SamplingBasedController

from franka_panda_server import FrankaPandaServer


def _assert_gpu_available(node_logger=None):
    devs = jax.devices()
    has_gpu = any(d.platform == "gpu" for d in devs)
    msg = f"JAX backend: {jax.default_backend()} | devices: {[str(d) for d in devs]}"
    if node_logger:
        node_logger.info(msg)
    else:
        print(msg)
    if not has_gpu:
        warn = ("No GPU devices visible to JAX. "
                "Check that jaxlib is the CUDA build and CUDA drivers/CUDA_VISIBLE_DEVICES are set.")
        if node_logger:
            node_logger.error(warn)
        else:
            print(warn)
    return has_gpu


class FR3_PushT(FrankaPandaServer):
    """Planner runs in its own thread on GPU; ROS2 servo republishes latest action."""
    def __init__(self, ctrl: SamplingBasedController, robot_ip: str, seed: int):
        super().__init__(robot_ip, None)

        # ----------------- Scene safety -----------------
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

        # ----------------- TF setup -----------------
        self.br = StaticTransformBroadcaster(self)
        self._publish_static_robot_tf()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----------------- Controller / JAX -----------------
        self.ctrl = ctrl
        # Build mjx data from task model; keep on device
        self.mjx_data = mjx.forward(ctrl.task.model, mjx.make_data(ctrl.task.model))
        # Initialize controller params on device BEFORE starting planner
        self.policy_params = device_put(ctrl.init_params(seed))

        # Validate GPU visibility
        _assert_gpu_available(self.get_logger())

        # JIT compile optimize -> returns new params; donate params to keep them on device
        self.get_logger().info("Jitting controller optimize() for GPU...")
        t0 = time.time()
        self._jit_optimize = jax.jit(lambda d, p: ctrl.optimize(d, p)[0], donate_argnums=(1,))
        # Compile intentionally to current shapes/devices
        self._executable = self._jit_optimize.lower(self.mjx_data, self.policy_params).compile()
        self._get_action = ctrl.get_action
        t1 = time.time()
        self.get_logger().info(f"JIT compile finished in {t1 - t0:.2f} s")

        # ----------------- Buffers -----------------
        nq, nv = self.mjx_data.qpos.shape[0], self.mjx_data.qvel.shape[0]
        self._tmp_lin = np.empty(3, dtype=np.float64)
        self._tmp_quat = np.empty(4, dtype=np.float64)
        self._idx_robot_start, self._idx_robot_end = 3, -2
        # Keep device copies; we will update with JAX ops to avoid host fallback
        self._q_dev  = jnp.array(self.mjx_data.qpos)
        self._dq_dev = jnp.array(self.mjx_data.qvel)

        # Command/action state (host)
        self._current_u = np.zeros((2,), dtype=np.float32)

        # Logging
        self._servo_tick = 0
        self._planner_tick = 0
        self._log_every = 100
        self._last_servo_time = time.time()
        self._last_plan_time = time.time()

        # ----------------- Timers & threading -----------------
        self.plan_max_hz = 100.0  # cap; 0/None = uncapped
        self.servo_freq  = 50.0

        self.servo_group = ReentrantCallbackGroup()

        # Servo setup
        self.get_logger().info("Starting servo...")
        self.servo.enable_servo()
        self.servo.use_twist()

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        self.create_timer(1.0/self.servo_freq, self._send_command, callback_group=self.servo_group)

        # Dedicated planner thread (start AFTER everything above is initialized)
        self._planner_stop = False
        self._planner_thread = threading.Thread(target=self._planner_loop, name="planner", daemon=True)
        self._planner_thread.start()
        self.get_logger().info("Planner thread started.")

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

    # ---------- State update helpers ----------
    def _get_T_pose(self):
        if not self.tf_buffer.can_transform("fr3_link0", "objectPushT",
                                            rclpy.time.Time(),
                                            rclpy.duration.Duration(seconds=0.01)):
            return None
        t = self.tf_buffer.lookup_transform("fr3_link0", "objectPushT", rclpy.time.Time())
        tr, rq = t.transform.translation, t.transform.rotation
        self._tmp_lin[:]  = (tr.x, tr.y - 0.025, tr.z)  # small offset as in your original
        self._tmp_quat[:] = (rq.x, rq.y, rq.z, rq.w)
        return self._tmp_lin, self._tmp_quat

    def _update_state(self):
        tvals = self._get_T_pose()
        if tvals is None:
            return False
        lin_t, quat_t = tvals

        ang = math.atan2(2*(quat_t[3]*quat_t[2] + quat_t[0]*quat_t[1]),
                         1 - 2*(quat_t[1]*quat_t[1] + quat_t[2]*quat_t[2])) - math.pi

        # Use pure JAX ops so updates remain on device (no CPU bounce)
        q, dq = self._q_dev, self._dq_dev
        q = q.at[0].set(-lin_t[1])
        q = q.at[1].set(lin_t[0] - 0.44)
        q = q.at[2].set(ang)

        js = getattr(self, "_current_joint_state", None)
        if js is None:
            return False
        s, e = self._idx_robot_start, self._idx_robot_end
        q  = q.at[s:e].set(jnp.asarray(js.position, dtype=q.dtype))
        dq = dq.at[s:e].set(jnp.asarray(js.velocity, dtype=dq.dtype))

        self._q_dev, self._dq_dev = q, dq
        self.mjx_data = self.mjx_data.replace(qpos=q, qvel=dq)
        return True

    # ---------- Planner loop (GPU) ----------
    def _planner_loop(self):
        # Frequency cap
        period = 0.0
        try:
            if self.plan_max_hz and self.plan_max_hz > 0:
                period = 1.0 / float(self.plan_max_hz)
        except Exception:
            period = 0.0

        next_deadline = time.perf_counter()

        while rclpy.ok() and not self._planner_stop:
            if self._update_state():
                try:
                    # Run compiled optimize step on GPU; donate params to keep on device
                    new_params = self._executable(self.mjx_data, self.policy_params)
                    # Optionally synchronize once in a while to surface backend early
                    # new_params = jax.block_until_ready(new_params)

                    # Read tiny action to host for the servo
                    act = self._get_action(new_params, 0.0)

                    # Commit
                    self.policy_params = new_params
                    self._current_u = np.asarray(act, dtype=np.float32)

                    # Log sometimes
                    self._planner_tick += 1
                    if self._planner_tick % self._log_every == 0:
                        now = time.time()
                        dt = now - self._last_plan_time
                        if dt > 0:
                            freq = self._log_every / dt
                            self.get_logger().info(f"[Planner] {freq:.1f} Hz | backend={jax.default_backend()}")
                        self._last_plan_time = now
                except Exception as e:
                    self.get_logger().error(f"Planner error (continuing): {e}")

            # soft rate cap
            if period > 0.0:
                next_deadline += period
                sleep_for = next_deadline - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_deadline = time.perf_counter()

        self.get_logger().info("Planner thread exiting.")

    # ---------- Servo publisher ----------
    def _send_command(self):
        u = None #getattr(self, "_current_u", None)
        if u is None or (hasattr(u, "size") and u.size < 2):
            self.servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
        else:
            self.servo(linear=(float(u[0]), float(u[1]), 0.0),
                       angular=(0.0, 0.0, 0.0))

        self._servo_tick += 1
        if self._servo_tick % self._log_every == 0:
            now = time.time()
            dt = now - self._last_servo_time
            if dt > 0:
                freq = self._log_every / dt
                self.get_logger().info(f"[Servo] {freq:.1f} Hz")
            self._last_servo_time = now


# --------- Main ---------
if __name__ == "__main__":
    rclpy.init()

    parser = argparse.ArgumentParser(description="FR3 Push-T threaded GPU planner")
    parser.add_argument("--algorithm", choices=["mppi", "mtp"], default="mtp")
    parser.add_argument("--plan-max-hz", type=float, default=100.0)
    parser.add_argument("--cuda-device", type=str, default=None, help="e.g. 0 or 0,1")
    args = parser.parse_args()

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Task + controller
    task = PushTFranka(
        ik_type='pinv',
        planning_horizon=10,
        sim_steps_per_control_step=2,
        ctrl_limits={
            "u_min": jnp.array([-0.35, -0.35]),
            "u_max": jnp.array([0.35, 0.35])
        },
        trace_sites=[],
        actuation_type='velocity',
    )
    seed = 42
    if args.algorithm == "mppi":
        ctrl = MPPI(task,
                    num_samples=512,
                    noise_level=0.2,
                    temperature=0.1,
                    num_randomizations=5,
                    seed=seed)
    else:
        ctrl = MTP(task,
                   num_samples=64,
                   M=2, N=16,
                   num_elites=12,
                   beta=0.25,
                   alpha=0.01,
                   interpolation='bspline',
                   num_randomizations=5,
                   seed=seed)

    node = FR3_PushT(ctrl=ctrl, robot_ip="10.90.90.77", seed=seed)
    node.plan_max_hz = args.plan_max_hz

    # Print devices once more in main
    _assert_gpu_available(node.get_logger())

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        try:
            node._planner_stop = True
            if hasattr(node, "_planner_thread"):
                node._planner_thread.join(timeout=2.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()