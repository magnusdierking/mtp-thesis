import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped
from sensor_msgs.msg import JointState


from pymoveit2 import MoveIt2
from pymoveit2.robots import panda
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile


class FR3Robot(Node):
    
    
    def __init__(self):
        super().__init__("fr3_robot")

        # Create callback group that allows execution of callbacks in parallel without restrictions
        self._callback_group = ReentrantCallbackGroup()
        
        self.joint_names = panda.joint_names()
        self.base_link_name = panda.base_link_name()
        self.end_effector_name = panda.end_effector_name()
        self.move_group_name = panda.MOVE_GROUP_ARM

        self._home_configuration = { "fr3_arm" : {
            "fr3_joint_1": 0.0,
            "fr3_joint_2": -np.pi/4,
            "fr3_joint_3": 0.0,
            "fr3_joint_4": -3*np.pi/4,
            "fr3_joint_5": 0.0,
            "fr3_joint_6": np.pi/2,
            "fr3_joint_7": np.pi/4
        }}
        
        # ----------------------------------------------
        # MoveIt 2 setup
        # ----------------------------------------------
        
        # Create MoveIt 2 interface
        self._moveit2 = MoveIt2(
            node=self,
            joint_names=self.joint_names,
            base_link_name=self.base_link_name,
            end_effector_name=self.end_effector_name,
            group_name=self.move_group_name,
            execute_via_moveit=True,
            callback_group=self._callback_group,
        )
        # Use upper joint velocity and acceleration limits
        self._moveit2.max_velocity = 1.0
        self._moveit2.max_acceleration = 1.0
        
        
        # ----------------------------------------------
        # Publishers and Subscribers
        # ----------------------------------------------

        self._current_pose_stamped = None
        self.create_subscription(
            PoseStamped,
            "/franka_robot_state_broadcaster/current_pose",
            self._ee_pose_callback,
            QoSProfile(depth=1),
            callback_group=self._callback_group,
        )
        
        self._current_joint_state = None
        self.create_subscription(
            JointState,
            "/franka_robot_state_broadcaster/current_joint_state",
            self._joint_state_callback,
            QoSProfile(depth=1),
            callback_group=self._callback_group,
        )




    # ----------------------------------------------
    # Callbacks and Helper Methods
    # ----------------------------------------------
    
    def _ee_pose_callback(self, msg: PoseStamped):
        """
        Callback for the end-effector pose.
        This is where you can handle the incoming pose data.
        """
        self.get_logger().info(f"Received end-effector pose: {msg.pose}")
        self._current_pose_stamped = msg
    
    def _extract_current_pose(self):
        """
        Extract the current pose from the stored PoseStamped message.
        """
        if self._current_pose_stamped is not None:
            pos = np.array([self._current_pose_stamped.pose.position.x,
                            self._current_pose_stamped.pose.position.y,
                            self._current_pose_stamped.pose.position.z])
            quat_xyzw = np.array([self._current_pose_stamped.pose.orientation.x,
                                  self._current_pose_stamped.pose.orientation.y,
                                  self._current_pose_stamped.pose.orientation.z,
                                  self._current_pose_stamped.pose.orientation.w])
            return pos, quat_xyzw
        else:
            self.get_logger().warn("Current pose not available.")
            return None
        
    def _joint_state_callback(self, msg: JointState):
        """
        Callback for the joint state.
        This is where you can handle the incoming joint state data.
        """
        self.get_logger().info(f"Received joint state: {msg.position}")
        self._current_joint_state = msg
        
    def _extract_current_joint_state(self):
        """
        Extract the current joint state from the stored JointState message.
        """
        joint_data = {}
        msg = self._current_joint_state
        if msg is None:
            self.get_logger().warn("Current joint state not available.")
            return None
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                joint_data[name] = {
                    "position": msg.position[i] if i < len(msg.position) else None,
                    "velocity": msg.velocity[i] if i < len(msg.velocity) else None,
                    "effort": msg.effort[i] if i < len(msg.effort) else None,
                }
        return joint_data


        
        
        
    # ----------------------------------------------
    # MoveIt 2 Interface Methods
    # ----------------------------------------------
    
    def plan_and_move_to_pose(self, target_pose):
        """
        Plan and move the robot to a specified pose.
        """
        if self._moveit2 is None:
            self.get_logger().error("MoveIt2 interface not initialized.")
            return
        if target_pose.shape != (4, 4) or not np.allclose(target_pose[3, :], [0, 0, 0, 1]) or not np.isclose(np.linalg.det(target_pose[:3, :3]), 1.0):
            raise ValueError("Pose must be a valid 4x4 homogeneous transformation matrix.")
        
        self.get_logger().info(f"Planning to move to pose: {target_pose}")

        position = target_pose[:3, 3]
        rotation_matrix = target_pose[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        quat_xyzw = rotation.as_quat()  # gives xyzw by default
        
        self._moveit2.wait_until_executed()
        success = self._moveit2.move_to_pose(position=position, orientation=quat_xyzw)
        
        if success:
            self.get_logger().info("Move to pose successful.")
        else:
            self.get_logger().error("Failed to move to pose.")
            
            
    def plan_and_move_to_configuration(self, joint_positions):
        """
        Plans and moves the robot to specified joint positions via moveit2.
        """
        self.get_logger().info(f"Planning to move to joint positions: {joint_positions}")
        if self._moveit2 is None:
            self.get_logger().error("MoveIt2 interface not initialized.")
            return
        if len(joint_positions) != len(self.joint_names):
            raise ValueError(f"Expected {len(self.joint_names)} joint positions, got {len(joint_positions)}.")
        if not all(isinstance(pos, (int, float)) for pos in joint_positions):
            raise ValueError("All joint positions must be numeric (int or float).")
        
        success = self._moveit2.move_to_configuration(joint_positions)
        self._moveit2.wait_until_executed()
        if success:
            self.get_logger().info("Move to configuration successful.")
        else:
            self.get_logger().error("Failed to move to configuration.")
        
        
    def move_to_home(self):
        """ 
        Moves the robot to its home configuration.
        """
        joint_positions = list(self._home_configuration["fr3_arm"].values())
        
        self.get_logger().info(
            f"Moving to home configuration: {joint_positions}"
        )
        self.plan_and_move_to_configuration(joint_positions)
        
    
    # ----------------------------------------------
    # MoveIt 2 Collision 
    # ----------------------------------------------