from copy import deepcopy
from typing import Optional, Tuple

from geometry_msgs.msg import TwistStamped
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
import rclpy
from rclpy.task import Future
from std_srvs.srv import SetBool
from moveit_msgs.srv import ServoCommandType


class MoveIt2Servo:
    """
    Python interface for MoveIt 2 Servo that enables real-time control in Cartesian Space.
    This implementation is just a thin wrapper around TwistStamped message publisher.
    """

    def __init__(
        self,
        node: Node,
        frame_id: str,
        namespace: str = "",
        linear_speed: float = 1.0,
        angular_speed: float = 1.0,
        enable_at_init: bool = True,
        callback_group: Optional[CallbackGroup] = None,
    ):
        """
        Construct an instance of `MoveIt2Servo` interface.
          - `node` - ROS 2 node that this interface is attached to
          - `namespace` - Namespace for the node
          - `frame_id` - Reference frame in which to publish command messages
          - `linear_speed` - Factor that can be used to scale all input linear twist commands
          - `angular_speed` - Factor that can be used to scale all input angular twist commands
          - `enable_at_init` - Flag that enables initialisation of MoveIt 2 Servo during initialisation
                               Otherwise, `MoveIt2Servo.enable()` must be called explicitly
          - `callback_group` - Optional callback group to use for ROS 2 communication (topics/services/actions)
        """

        self._node = node

        self.namespace = namespace

        # Create publisher
        self.__twist_pub = self._node.create_publisher(
            msg_type=TwistStamped,
            topic="/servo_node/delta_twist_cmds",
            qos_profile=QoSProfile(
                durability=QoSDurabilityPolicy.VOLATILE,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_ALL,
            ),
            callback_group=callback_group,
        )

        # Create service client (single pause/unpause service)
        self.__pause_service = self._node.create_client(
            srv_type=SetBool,
            srv_name=f"{self.namespace}/servo_node/pause_servo",
            callback_group=callback_group,
        )
        
        self.__switch_cmd_type_service = self._node.create_client(
            srv_type=ServoCommandType,
            srv_name=f"{self.namespace}/servo_node/switch_command_type",
            callback_group=callback_group,
        )

        # Track state as before
        self.__is_enabled = False
        self.__current_cmd_type = None  # one of JOINT_JOG/TWIST/POSE

        # Initialize message based on passed arguments
        self.__twist_msg = TwistStamped()
        self.__twist_msg.header.frame_id = frame_id
        self.__twist_msg.twist.linear.x = linear_speed
        self.__twist_msg.twist.linear.y = linear_speed
        self.__twist_msg.twist.linear.z = linear_speed
        self.__twist_msg.twist.angular.x = angular_speed
        self.__twist_msg.twist.angular.y = angular_speed
        self.__twist_msg.twist.angular.z = angular_speed

        # Enable servo immediately, if desired
        if enable_at_init:
            self.enable_servo()
            self.use_twist()  # default to twist commands

    
    def _call_pause(self, paused: bool, timeout_sec: float = 2.0):
        # Make sure the service is up
        if not self.__pause_service.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().error(f"pause_servo service with namespace {self.namespace} not available")
            return False, "Service unavailable"

        req = SetBool.Request()
        req.data = paused  # True -> pause (disable), False -> unpause (enable)

        future = self.__pause_service.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_sec)
        if not future.done():
            return False, "Service call timed out"

        resp = future.result()
        if resp is None:
            return False, "No response"

        # Keep your local flag consistent with the request/result
        if resp.success:
            self.__is_enabled = not paused

        return bool(resp.success), resp.message

    def enable_servo(self):
        """Enable (unpause) Servo."""
        ok, msg = self._call_pause(paused=False)
        if not ok:
            self._node.get_logger().warn(f"Failed to enable Servo: {msg}")
        return ok, msg

    def disable_servo(self):
        """Disable (pause) Servo."""
        ok, msg = self._call_pause(paused=True)
        if not ok:
            self._node.get_logger().warn(f"Failed to disable Servo: {msg}")
        return ok, msg  

    def set_servo_command_type(self, cmd_type: int, timeout_sec: float = 2.0) -> bool:
        """
        cmd_type: ServoCommandType.Request.JOINT_JOG | TWIST | POSE
        """
        if not self.__switch_cmd_type_service.wait_for_service(timeout_sec=timeout_sec):
            self._node.get_logger().error("switch_command_type service not available")
            return False

        req = ServoCommandType.Request()
        req.command_type = cmd_type

        fut = self.__switch_cmd_type_service.call_async(req)
        rclpy.spin_until_future_complete(self._node, fut, timeout_sec=timeout_sec)
        resp = fut.result()
        if not resp or not resp.success:
            self._node.get_logger().warn("Failed to switch Servo command type")
            return False

        self.__current_cmd_type = cmd_type
        return True
    
    def use_joint_jog(self):  # publish to /servo_node/delta_joint_cmds
        return self.set_servo_command_type(ServoCommandType.Request.JOINT_JOG)

    def use_twist(self):      # publish to /servo_node/delta_twist_cmds
        return self.set_servo_command_type(ServoCommandType.Request.TWIST)

    def use_pose(self):       # publish to /servo_node/pose_target_cmds
        return self.set_servo_command_type(ServoCommandType.Request.POSE)


    def __del__(self):
        """
        Try to stop MoveIt 2 Servo during destruction.
        """

        try:
            if self.is_enabled:
                self.__stop_service.call_async(self.__trigger_req)
        except:
            pass

    def __call__(
        self,
        linear: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Callable that is identical to `MoveIt2Servo.servo()`.
        """

        self.servo(linear=linear, angular=angular)

    def servo(
        self,
        linear: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        enable_if_disabled: bool = False,
    ):
        """
        Apply linear and angular twist using MoveIt 2 Servo.
        Input is scaled by `linear_speed` and `angular_speed`, respectively.
        """

        if not self.is_enabled:
            self._node.get_logger().warn(
                "Command failed because MoveIt Servo is not yet enabled."
            )
            if enable_if_disabled:
                # self._node.get_logger().warn(
                #     f"Calling '{self.__start_service.srv_name}' service to enable MoveIt Servo..."
                # )
                # if not self.enable():
                #     return
                pass
            else:
                return

        twist_msg = deepcopy(self.__twist_msg)
        twist_msg.header.stamp = self._node.get_clock().now().to_msg()
        twist_msg.twist.linear.x *= linear[0]
        twist_msg.twist.linear.y *= linear[1]
        twist_msg.twist.linear.z *= linear[2]
        twist_msg.twist.angular.x *= angular[0]
        twist_msg.twist.angular.y *= angular[1]
        twist_msg.twist.angular.z *= angular[2]
        self.__twist_pub.publish(twist_msg)

    # def enable(
    #     self, wait_for_server_timeout_sec: Optional[float] = 1.0, sync: bool = False
    # ) -> bool:
    #     """
    #     Enable MoveIt 2 Servo server via async service call.
    #     """

    #     while not self.__start_service.wait_for_service(
    #         timeout_sec=wait_for_server_timeout_sec
    #     ):
    #         self._node.get_logger().warn(
    #             f"Service '{self.__start_service.srv_name}' is not yet available..."
    #         )
    #         return False

    #     if sync:
    #         result = self.__start_service.call(self.__trigger_req)
    #         if not result.success:
    #             self._node.get_logger().error(
    #                 f"MoveIt Servo could not be enabled. ({result.message})"
    #             )
    #         self.__is_enabled = result.success
    #         return result.success
    #     else:
    #         start_service_future = self.__start_service.call_async(self.__trigger_req)
    #         start_service_future.add_done_callback(self.__enable_done_callback)
    #         return True

    # def disable(
    #     self, wait_for_server_timeout_sec: Optional[float] = 1.0, sync: bool = False
    # ) -> bool:
    #     """
    #     Disable MoveIt 2 Servo server via async service call.
    #     """

    #     while not self.__stop_service.wait_for_service(
    #         timeout_sec=wait_for_server_timeout_sec
    #     ):
    #         self._node.get_logger().warn(
    #             f"Service '{self.__stop_service.srv_name}' is not yet available..."
    #         )
    #         return False

    #     if sync:
    #         result = self.__stop_service.call(self.__trigger_req)
    #         if not result.success:
    #             self._node.get_logger().error(
    #                 f"MoveIt Servo could not be disabled. ({result.message})"
    #             )
    #         self.__is_enabled = not result.success
    #         return result.success
    #     else:
    #         stop_service_future = self.__stop_service.call_async(self.__trigger_req)
    #         stop_service_future.add_done_callback(self.__disable_done_callback)
    #         return True

    # def __enable_done_callback(self, future: Future):
    #     result: Trigger.Response = future.result()

    #     if not result.success:
    #         self._node.get_logger().error(
    #             f"MoveIt Servo could not be enabled. ({result.message})"
    #         )

    #     self.__is_enabled = result.success

    # def __disable_done_callback(self, future: Future):
    #     result: Trigger.Response = future.result()

    #     if not result.success:
    #         self._node.get_logger().error(
    #             f"MoveIt Servo could not be disabled. ({result.message})"
    #         )

    #     self.__is_enabled = not result.success

    @property
    def is_enabled(self) -> bool:
        return self.__is_enabled

    @property
    def frame_id(self) -> str:
        return self.__twist_msg.header.frame_id

    @frame_id.setter
    def frame_id(self, value: str):
        self.__twist_msg.header.frame_id = value

    @property
    def linear_speed(self) -> float:
        return self.__twist_msg.twist.linear.x

    @linear_speed.setter
    def linear_speed(self, value: float):
        self.__twist_msg.twist.linear.x = value
        self.__twist_msg.twist.linear.y = value
        self.__twist_msg.twist.linear.z = value

    @property
    def angular_speed(self) -> float:
        return self.__twist_msg.twist.angular.x

    @angular_speed.setter
    def angular_speed(self, value: float):
        self.__twist_msg.twist.angular.x = value
        self.__twist_msg.twist.angular.y = value
        self.__twist_msg.twist.angular.z = value
