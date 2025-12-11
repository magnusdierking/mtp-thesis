from abc import ABC, abstractmethod
from threading import Thread

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import rclpy

from pymoveit2 import MoveIt2Servo
from pymoveit2 import MoveIt2
from pymoveit2.robots import panda as robot

from tf2_ros import Buffer, TransformListener

class RobotServer(ABC, Node):
    def __init__(self, 
                 robot_name,
                 robot_ip, 
                 gripper_type, 
                 robot_prefix,
                 moveit_group_name):
        super().__init__(robot_name)  # This makes this class a ROS Node
        
        self._robot_ip = robot_ip
        self._gripper_type = gripper_type
        self._robot_prefix = robot_prefix
        self._moveit_group_name = moveit_group_name
        
        self._home_configuration = None
        # to be set by callbacks
        self._current_pose = None
        self._current_twist = None
        self._current_joint_state = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Create MoveIt2 interface
        self._callback_group = ReentrantCallbackGroup()  
        self._moveit2 = MoveIt2(
            node=self,  
            joint_names=robot.joint_names(prefix=robot_prefix),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(prefix=robot_prefix),
            group_name=self._moveit_group_name,
            callback_group=self._callback_group,
        )
        self.servo = MoveIt2Servo(
            node=self,
            linear_speed=1.0,
            angular_speed=1.0,
            frame_id=robot.base_link_name(),
            callback_group=self._callback_group,
        )
    

    @abstractmethod
    def plan_and_move_to_pose(self):
        pass

    @abstractmethod
    def plan_and_move_to_configuration(self, joint_positions):
        pass

    @abstractmethod
    def move_to_home(self):
        pass

    @abstractmethod
    def get_joint_positions(self):
        pass
        
    @abstractmethod
    def get_joint_velocities(self):
        pass

    @abstractmethod
    def get_joint_states(self):
        pass

    @abstractmethod
    def get_ee_pose(self, frame : str = "base_link"): 
        pass

    @abstractmethod
    def get_ee_twist(self, frame : str = "base_link"):
        pass
