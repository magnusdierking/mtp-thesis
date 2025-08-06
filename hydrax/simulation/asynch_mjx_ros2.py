import multiprocessing as mp
import time
from multiprocessing import Event, shared_memory, Condition
from math import cos, sin
import pprint


import rclpy
import numpy as np
import mujoco
from robot_interfaces.robots.panda_bridge import PandaBridge
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive


from hydrax.alg_base import SamplingBasedController


class SharedMemoryNumpyArray:
    """Helper class to store a numpy array in shared memory."""

    def __init__(self, arr: np.ndarray, ctx: mp.context.BaseContext):
        """Create a shared memory numpy array.

        Args:
            arr: The numpy array to store in shared memory. Size and dtype must
                 be fixed.
            ctx: The multiprocessing context to use for shared memory.
        """
        self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
        shared_arr[:] = arr[:]
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.lock = ctx.Lock()

    def __getitem__(self, key: int) -> np.ndarray:
        """Get an item from the shared array."""
        shm = shared_memory.SharedMemory(name=self.shm.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return np.copy(arr[key])  # Need to copy here to avoid segfaults

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        """Set an item in the shared array."""
        with self.lock:
            shm = shared_memory.SharedMemory(name=self.shm.name)
            arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
            arr[key] = value

    def __str__(self) -> str:
        """Return the string representation of the shared array."""
        shm = shared_memory.SharedMemory(name=self.shm.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return str(arr)

    def __del__(self) -> None:
        """Clean up the shared memory on deletion."""
        self.shm.close()
        self.shm.unlink()
        
    def as_numpy(self) -> np.ndarray:
        """Return a full copy of the shared memory as a NumPy array."""
        shm = shared_memory.SharedMemory(name=self.shm.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return arr.copy()  # Important: copy to avoid shared memory issues



class SharedMemoryMujocoData:
    """Helper class for passing mujoco data between concurrent processes."""

    def __init__(self, mj_data: mujoco.MjData, ctx: mp.context.BaseContext):
        """Create shared memory objects for state and control data.

        Note that this does not copy the full mj_data object, only those fields
        that we want to share between the simulator and controller.

        Args:
            mj_data: The mujoco data object to store in shared memory.
            ctx: The multiprocessing context to use.
        """
        # store flag whether all real states have been set, 
        # used to signal that the controller can start
        self.states_received = ctx.Value("b", False)

        # N.B. we use float32 to match JAX's default precision
        self.qpos = SharedMemoryNumpyArray(
            np.array(mj_data.qpos, dtype=np.float32), ctx
        )
        self.qvel = SharedMemoryNumpyArray(
            np.array(mj_data.qvel, dtype=np.float32), ctx
        )
        self.ctrl = SharedMemoryNumpyArray(
            np.array(mj_data.ctrl, dtype=np.float32), ctx
        )

        # if len(mj_data.mocap_pos) > 0:
        #     self.mocap_pos = SharedMemoryNumpyArray(
        #         np.array(mj_data.mocap_pos, dtype=np.float32), ctx
        #     )
        #     self.mocap_quat = SharedMemoryNumpyArray(
        #         np.array(mj_data.mocap_quat, dtype=np.float32), ctx
        #     )



def run_controller(
    ctrl: SamplingBasedController,
    shm_data: SharedMemoryMujocoData,
    finished: Event,
    jitted: Event,
    cond_receiving_states: Condition,
    seed: int = 0,
) -> None:
    """Run the controller, communicating with the simulator over shared memory.

    Args:
        ctrl: The controller instance (which includes the task definition).
        shm_data: Shared memory object for state and control action data.
        ready: Shared flag for signaling that the controller is ready.
        finished: Shared flag for stopping the simulation.
    """
    
    import jax
    import jax.numpy as jnp
    from jax import tree_util
    from mujoco import mjx
    
    def load_from_shared(mjx_data, shm_data, clamp=True, clean=True):
        qpos_np = shm_data.qpos.as_numpy()
        qvel_np = shm_data.qvel.as_numpy()

        updated = mjx_data.replace(
            qpos=jax.device_put(jnp.array(qpos_np, dtype=jnp.float32)),
            qvel=jax.device_put(jnp.array(qvel_np, dtype=jnp.float32)),
        )

        if hasattr(shm_data, "mocap_pos") and len(mjx_data.mocap_pos) > 0:
            mocap_pos_np = shm_data.mocap_pos.as_numpy()
            mocap_quat_np = shm_data.mocap_quat.as_numpy()

            updated = updated.replace(
                mocap_pos=jax.device_put(jnp.array(mocap_pos_np, dtype=mjx_data.mocap_pos.dtype)),
                mocap_quat=jax.device_put(jnp.array(mocap_quat_np, dtype=mjx_data.mocap_quat.dtype)),
            )

        if clean:
            updated = jax.tree_util.tree_map(
                lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6), updated
            )

        if clamp:
            updated = updated.replace(
                qpos=jnp.clip(updated.qpos, -1e6, 1e6),
                qvel=jnp.clip(updated.qvel, -1e6, 1e6),
            )

        pprint.pprint("Loaded data from shared memory:")
        pprint.pprint(f"qpos: {updated.qpos}")
        pprint.pprint(f"qvel: {updated.qvel}")
        pprint.pprint(f"ctrl: {shm_data.ctrl.as_numpy()}")

        return updated
    
    # Initialize the policy parameters and state estimate
    mjx_data = mjx.make_data(ctrl.task.model)
    policy_params = ctrl.init_params(seed)

    # Print out some planning horizon information
    print(
        f"Planning with {ctrl.task.planning_horizon} steps "
        f"over a {ctrl.task.planning_horizon * ctrl.task.dt} second horizon."
    )

    while not shm_data.states_received.value:
        # Wait until we have received the initial state from the actual robot
        time.sleep(0.1)
    # with cond_receiving_states:
    #     print("Waiting for shared state object to be initialized ...")
    #     cond_receiving_states.wait()
    #     print("Shared state object initialized, starting controller...")
    
    
    # Jit the optimizer step, then signal that we're ready to go
    print("Jitting controller...")
    print("This may take a while, please be patient.")
    st = time.time()
    # Use raw Python functions instead of JIT-compiled versions
    # jit_optimize = lambda d, p: ctrl.optimize(d, p)[0]
    # get_action = ctrl.get_action
    
    mjx_data = load_from_shared(mjx_data, shm_data, clamp=False, clean=False)
    mjx_data = mjx.forward(ctrl.task.model, mjx_data)
    def summarize(x):
        return {
            "shape": x.shape,
            "dtype": x.dtype,
            "nan": jnp.isnan(x).any(),
            "inf": jnp.isinf(x).any(),
            "norm": jnp.linalg.norm(x)
        }

    summary = tree_util.tree_map(summarize, mjx_data)
    pprint.pprint(summary)
    
    # opt_result = ctrl.optimize(mjx_data, policy_params)
    # print("First optimize result OK")
    # policy_params = opt_result[0]

    # policy_params = jit_optimize(mjx_data, policy_params)
    jit_optimize = jax.jit(
        lambda d, p: ctrl.optimize(d, p)[0], donate_argnums=(1,)
    )
    get_action = jax.jit(ctrl.get_action)
    policy_params = jit_optimize(mjx_data, policy_params)
    print(f"Time to jit: {time.time() - st}")
    jitted.set()

    
    while True: # Wait until we have received the initial state from the actual robot
        st = time.time()

        mjx_data = load_from_shared(mjx_data, shm_data, clamp=False, clean=False)
        print("Controller is ready, starting planning...")

        # Do a planning step
        policy_params = jit_optimize(mjx_data, policy_params)

        print("Planning step done, sending action to simulator...")
        # Send the action to the simulator.
        # TODO: send the full parameters rather than assuming zero-order
        # hold and a sufficiently high control rate
        a = get_action(policy_params, 0.0)
        print(f"Action: {a}")
        # shm_data.ctrl[:] = np.array(
        #     a, dtype=np.float32
        # )
        freq = 1 / (time.time() - st)
        print(f"Controller running at {freq:.3f} Hz")


def run_ros2_interface(
    shm_data: SharedMemoryMujocoData,
    finished: Event,
    jitted: Event,
    cond_receiving_states: Condition,
) -> None:
    """Run a simulation, communicating with the controller over shared memory.

    Args:
        mj_model: Mujoco model for the simulation.
        mj_data: Mujoco data specifying the initial state.
        shm_data: Shared memory object for state and control action data.
        ready: Shared flag for starting the simulation.
        finished: Shared flag for stopping the simulation.
        delay_ctrl_start: Whether to delay the controller start.
    """

    rclpy.init()
    
    # # wait until the controller is ready (jitted)
    # while not jitted.is_set():
    #     # wait for the controller to be ready
    #     #print("Waiting for controller to be ready...")
    #     time.sleep(0.1)
    
    robot = PandaBridge(shm_data,
                        robot_ip='10.90.90.144',
                        cond_receiving_states=cond_receiving_states)

    robot.add_collision_primitive(
        id="box1",
        primitive_type=SolidPrimitive.BOX,
        dimensions=(1, 1, 0.1),
        position=np.array([0.0, 0.0, -0.05]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
    )

    def servo_circular_motion():
        """Move in a circular motion using Servo"""

        now_sec = robot.get_clock().now().nanoseconds * 1e-9
        robot.servo(linear=(sin(now_sec), cos(now_sec), 0.0), angular=(0.0, 0.0, 0.0))
        # print(shm_data.qpos[:])
        
    #robot.create_timer(0.05, servo_circular_motion)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(robot)
    
    try:   
        executor.spin()
    finally:
        robot.destroy_node()
        shm_data.close()
        shm_data.unlink()
        finished.set()
        rclpy.shutdown()
        
        

def run_interactive(
    controller: SamplingBasedController,
    mj_data: mujoco.MjData,
    delay_ctrl_start: bool=False,
) -> None:
    """Run an asynchronous interactive simulation.

    This is similar to `simulation.deterministic.run_interactive`, but runs the
    controller and simulator in separate processes. This is more realistic, but
    offers fewer features (e.g., no trace visualization).

    Args:
        controller: The controller to use for planning.
        mj_data: Mujoco data specifying the initial state.
        delay_ctrl_start: Whether to delay the controller start.
    """
    ctx = mp.get_context("spawn")  # Need to use spawn for jax compatibility
    shm_data = SharedMemoryMujocoData(mj_data, ctx)
    # init the shared memory data with zeros
    shm_data.qpos[:] = np.zeros(mj_data.qpos.shape, dtype=np.float32)
    shm_data.qvel[:] = np.zeros(mj_data.qvel.shape, dtype=np.float32)
    shm_data.ctrl[:] = np.zeros(mj_data.ctrl.shape, dtype=np.float32)
    
    jitted = ctx.Event()
    finished = ctx.Event()
    
    # condition for 
    # a) controller to start planning
    # b) controller to keep running 
    cond_receiving_states = Condition()

    # Set up ros interface 
    sim = ctx.Process(
        target=run_ros2_interface,
        args=(shm_data, finished, jitted, cond_receiving_states),
    )
    control = ctx.Process(
        target=run_controller, 
        args=(controller, shm_data, finished, jitted, cond_receiving_states)
    )

    # Run the simulation and controller in parallel
    sim.start()
    control.start()

    # Clean up when done (e.g. the visualizer is closed)
    sim.join()
    control.join()
