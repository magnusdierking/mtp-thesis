"""
Minimal Servo Node — subscribes to action splines and sends MoveIt Servo
commands at high frequency.

This node is intentionally lightweight so it never blocks on heavy compute.
It runs in its own process, completely free of the GIL contention caused by
JAX planning in the planner node.

Subscribes:
    /smpc/action_spline  (std_msgs/Float32MultiArray)
        data = [dt, vx_0, vy_0, vx_1, vy_1, …, vx_{H-1}, vy_{H-1}]

Usage:
    python servo_node.py                    # SMPC mode (default)
    python servo_node.py --teleop           # start in teleop mode
    python servo_node.py --freq 50          # 50 Hz servo rate
"""

import argparse
import threading
import time

import numpy as np
import rclpy
from pymoveit2 import MoveIt2Servo
from pymoveit2.robots import panda as robot
from pynput import keyboard
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from pymoveit2 import MoveIt2Servo
from pymoveit2 import MoveIt2
from pymoveit2.robots import panda as robot

class ServoNode(Node):
    """
    Subscribes to ``/smpc/action_spline`` and indexes into the spline at a
    configurable servo rate (default 30 Hz).  Supports keyboard teleop as a
    fallback (toggle with 't').
    """

    def __init__(self, servo_freq: float = 30.0, start_in_teleop: bool = False):
        super().__init__("smpc_servo_node")
        self.get_logger().info(f"Initialising Servo Node at {servo_freq} Hz …")

        self.servo_freq = servo_freq

        # ----------------------------------------------------------
        #  MoveIt Servo interface
        # ----------------------------------------------------------
        self._cb_group = ReentrantCallbackGroup()
        self._servo = MoveIt2Servo(
            node=self,
            linear_speed=1.0,
            angular_speed=1.0,
            frame_id=robot.base_link_name(),
            callback_group=self._cb_group,
            enable_at_init=True,
        )
        time.sleep(1.0)  # wait for MoveIt Servo to be ready
        self._servo.use_twist()

        # ----------------------------------------------------------
        #  Spline subscription
        # ----------------------------------------------------------
        self._spline_lock = threading.Lock()
        self._actions: np.ndarray | None = None  # (H, 2)
        self._dt: float = 0.0  # seconds per index
        self._spline_stamp: float = 0.0  # wall-clock when received

        self.create_subscription(
            Float32MultiArray,
            "/smpc/action_spline",
            self._spline_callback,
            10,
            callback_group=self._cb_group,
        )

        # ----------------------------------------------------------
        #  Keyboard teleop
        # ----------------------------------------------------------
        self.velocity = 0.2  # m/s
        self._key_lock = threading.Lock()
        self.teleop_enabled = start_in_teleop
        self._key_vx = 0.0
        self._key_vy = 0.0

        threading.Thread(target=self._keyboard_loop, daemon=True).start()

        # ----------------------------------------------------------
        #  Servo timer
        # ----------------------------------------------------------
        self.create_timer(
            1.0 / self.servo_freq,
            self._servo_tick,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"Servo Node ready  (teleop={'ON' if self.teleop_enabled else 'OFF'})"
        )

    # --------------------------------------------------------------
    #  Spline callback
    # --------------------------------------------------------------

    def _spline_callback(self, msg: Float32MultiArray):
        """Unpack [dt, vx0, vy0, vx1, vy1, …] into an (H, 2) array."""
        data = msg.data
        if len(data) < 3:
            self.get_logger().warn("Received spline with too few elements, ignoring.")
            return
        dt = float(data[0])
        flat = np.array(data[1:], dtype=np.float32)
        nu = 2
        if flat.size % nu != 0:
            self.get_logger().warn(
                f"Spline payload size {flat.size} not divisible by {nu}, ignoring."
            )
            return
        actions = flat.reshape(-1, nu)

        with self._spline_lock:
            self._actions = actions
            self._dt = dt
            self._spline_stamp = time.time()

    # --------------------------------------------------------------
    #  Servo tick (high-frequency timer)
    # --------------------------------------------------------------

    def _servo_tick(self):
        if self.teleop_enabled:
            self._send_teleop()
            return

        # Read latest spline under lock
        with self._spline_lock:
            actions = self._actions
            dt = self._dt
            stamp = self._spline_stamp

        if actions is None or dt <= 0.0:
            # No plan yet — hold position
            self._servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
            return

        # Index into spline based on elapsed time since reception
        elapsed = time.time() - stamp
        idx = int(elapsed / dt)
        if idx >= len(actions):
            # Plan is exhausted — hold position
            self._servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
            return
        idx = max(0, min(idx, len(actions) - 1))

        vx = float(actions[idx, 0])
        vy = float(actions[idx, 1])

        self.get_logger().info(f"Servo tick: idx={idx}")

        self._servo(linear=(vx, vy, 0.0), angular=(0.0, 0.0, 0.0))

    # --------------------------------------------------------------
    #  Teleop
    # --------------------------------------------------------------

    def _send_teleop(self):
        with self._key_lock:
            vx = self._key_vx
            vy = self._key_vy
        self._servo(linear=(vx, vy, 0.0), angular=(0.0, 0.0, 0.0))

    def _keyboard_loop(self):
        """pynput listener that runs on a background thread."""

        def on_press(key):
            try:
                if key.char == "t":
                    self.teleop_enabled = not self.teleop_enabled
                    self.get_logger().info(f"teleop_enabled = {self.teleop_enabled}")
                    return
            except AttributeError:
                pass

            vx, vy = 0.0, 0.0
            if key == keyboard.Key.up:
                vx = self.velocity
            elif key == keyboard.Key.down:
                vx = -self.velocity
            elif key == keyboard.Key.left:
                vy = self.velocity
            elif key == keyboard.Key.right:
                vy = -self.velocity
            elif key == keyboard.Key.space:
                with self._key_lock:
                    self._key_vx = 0.0
                    self._key_vy = 0.0
                return
            else:
                return

            with self._key_lock:
                self._key_vx = vx
                self._key_vy = vy

        def on_release(key):
            if key in (
                keyboard.Key.up,
                keyboard.Key.down,
                keyboard.Key.left,
                keyboard.Key.right,
            ):
                with self._key_lock:
                    self._key_vx = 0.0
                    self._key_vy = 0.0

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


# ==================================================================
#  Main
# ==================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Minimal servo node for SMPC action splines.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=50.0,
        help="Servo command frequency in Hz (default: 50)",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Start in keyboard teleop mode",
    )
    args = parser.parse_args()

    rclpy.init()
    node = ServoNode(servo_freq=args.freq, start_in_teleop=args.teleop)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down servo node …")
    finally:
        # Send a zero-velocity stop command before exiting
        try:
            node._servo(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
        except Exception:
            pass
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
