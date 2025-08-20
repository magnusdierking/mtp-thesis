from os import path
import numpy as np
import time
import threading
from pprint import pformat
import trimesh
from copy import deepcopy
from typing import Tuple


import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from robot_interfaces.robots.robot_server import RobotServer

from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose


class FrankaPandaServer(RobotServer):
    def __init__(self, robot_ip, 
                       gripper_type):
        
        super().__init__("franka_panda_server", 
                         robot_ip, 
                         gripper_type, 
                         "fr3_",
                         "fr3_arm")  #     
        
        self._lock = threading.Lock()  # Lock for thread safety
        # Default home position from franka_ros2 xacro
        # https://github.com/frankarobotics/franka_ros2/blob/humble/franka_fr3_moveit_config/srdf/group_definition.xacro
        self._home_configuration = { "fr3_arm" : {
            "fr3_joint_1": 0.0,
            "fr3_joint_2": -np.pi/4,
            "fr3_joint_3": 0.0,
            "fr3_joint_4": -3*np.pi/4,
            "fr3_joint_5": 0.0,
            "fr3_joint_6": np.pi/2,
            "fr3_joint_7": np.pi/4
        }}
        
        self._ee_pose_subscriber = self.create_subscription(
            PoseStamped,
            "/franka_robot_state_broadcaster/current_pose",
            self._ee_pose_callback,
            10
        )

        self._ee_twist_subscriber = self.create_subscription(
            TwistStamped,
            "/franka_robot_state_broadcaster/current_twist",  
            self._ee_twist_callback,
            10
        )

        self._joint_state_subscriber = self.create_subscription(
            JointState,
            "/franka_robot_state_broadcaster/measured_joint_states",
            self._joint_state_callback,
            5
        )
            
        #self.create_timer(1.0, self._print_joint_states) 
        # For thesis
        #self.create_timer(0.01, self.update_states)  

    # Control

    def plan_and_move_to_pose(self, pose: np.ndarray):
        """
        Plans and moves the robot to a specified pose via moveit2.
        """
        # transform homogeneous transformation matrix to cartesian pose
        if pose.shape != (4, 4) or not np.allclose(pose[3, :], [0, 0, 0, 1]) or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0):
            raise ValueError("Pose must be a valid 4x4 homogeneous transformation matrix.")
        
        position = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        rotation = R.from_matrix(rotation_matrix)   
        quaternion = rotation.as_quat() # gives xyzw by default   
        
        self.get_logger().info(
            f"Moving to pose: {position} quat: {quaternion}"
        )
        self._moveit2.move_to_pose(
            position=position,
            quat_xyzw=quaternion,
            cartesian=False,
        )
        self._moveit2.wait_until_executed()


    def plan_and_move_to_configuration(self, joint_positions):
        """
        Plans and moves the robot to specified joint positions via moveit2.
        """
        self._moveit2.move_to_configuration(joint_positions)
        self._moveit2.wait_until_executed()


    def move_to_home(self):
        """
        Moves the robot to its home position via moveit2.
        """
        # Get the joint positions from the home configuration
        joint_positions = list(self._home_configuration["fr3_arm"].values())
        
        self.get_logger().info(
            f"Moving to home configuration: {joint_positions}"
        )
        self.plan_and_move_to_configuration(joint_positions)
    
    



    # Getter and Setter methods for state variables

    def get_joint_positions(self):
        """        
        Returns the current joint positions of the robot.
        """
        if not self._current_joint_state:
            return None
        else:
            # dictionary with joint names as keys and positions as values
            joint_positions = {}
            for i, name in enumerate(self._current_joint_state.name):
                joint_positions[name] = self._current_joint_state.position[i]
            return { "fr3_arm": joint_positions }
        
    def get_joint_velocities(self):
        """
        Returns the current joint velocities of the robot.
        """
        if not self._current_joint_state:
            return None
        else:
            # dictionary with joint names as keys and velocities as values
            joint_velocities = {}
            for i, name in enumerate(self._current_joint_state.name):
                joint_velocities[name] = self._current_joint_state.velocity[i]
            return { "fr3_arm": joint_velocities }
        
    def get_joint_efforts(self):
        """
        Returns the current joint efforts of the robot.
        """
        if not self._current_joint_state:
            return None
        else:
            # dictionary with joint names as keys and efforts as values
            joint_efforts = {}
            for i, name in enumerate(self._current_joint_state.name):
                joint_efforts[name] = self._current_joint_state.effort[i]
            return { "fr3_arm": joint_efforts }

    def get_joint_states(self):
        """
        Returns the current joint states of the robot.
        """
        return self.get_joint_positions(), self.get_joint_velocities(), self.get_joint_efforts()

    
    def get_ee_pose(self, frame : str = "base_link"): 
        # create homogeneous transformation matrix from pose
        if not self._current_pose:
            print("End-effector pose not available.")
            return None 
        else:
            if frame != "base_link":
                # Transform the pose to the specified frame
                print(f"Transforming end-effector pose to frame: {frame}")
                try:
                    transformed_pose = do_transform_pose(self._current_pose, self._tf_buffer.lookup_transform(frame, self._current_pose.header.frame_id, self._current_pose.header.stamp))
                except Exception as e:
                    print(f"Transform error: {e}")
                    return None
            else:
                transformed_pose = self._current_pose
            pose = transformed_pose.pose
            translation = np.array([pose.position.x, pose.position.y, pose.position.z])
            rotation = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
            rotation_matrix = rotation.as_matrix()
            H = np.eye(4)
            H[:3, :3] = rotation_matrix
            H[:3, 3] = translation
            return H
        
    def get_ee_twist(self, frame : str = "base"):
        pass


    ########################################################
    ##           Callbacks for ROS2 Subscribers           ##
    ########################################################
    
    def _ee_pose_callback(self, msg: PoseStamped):
        """
        Callback for end-effector pose updates.
        """
        # Process the end-effector pose message
        self._current_pose = deepcopy(msg)
        msg = None  # Clear the message to free memory
    
    def _ee_twist_callback(self, msg: TwistStamped):
        """
        Callback for end-effector twist updates.
        """
        # Process the end-effector twist message
        self._current_twist = deepcopy(msg)
        msg = None  # Clear the message to free memory
                
    def _joint_state_callback(self, msg):
        """ 
        Callback for joint configuration updates.
        """
        with self._lock:
            # Process the joint configuration message
            self._current_joint_state = deepcopy(msg)
            msg = None  # Clear the message to free memory


        
    def set_home_configuration(self, home_configuration):
        """
        Sets the home configuration for the robot.
        """
        if not isinstance(home_configuration, dict):
            raise ValueError("Home configuration must be a dictionary.")
        if "fr3_arm" not in home_configuration:
            raise ValueError("Home configuration must contain 'fr3_arm' key.")
        if home_configuration["fr3_arm"] is None:
            raise ValueError("Home configuration for 'fr3_arm' cannot be None.")
        if not all(isinstance(value, (int, float)) for value in home_configuration["fr3_arm"].values()):
            raise ValueError("All joint positions in home configuration must be numeric.")
        if len(home_configuration["fr3_arm"]) != 7:
            raise ValueError("Home configuration for 'fr3_arm' must contain exactly 7 joints.")
        
        self._home_configuration = home_configuration
        self.get_logger().info(f"Home configuration set: {pformat(self._home_configuration)}")
        
    def add_collision_mesh(self, 
                           mesh_path, 
                           frame_id="base_link", 
                           position=np.array([0.0, 0.0, 0.0]),
                           quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                           scale=np.array([1.0, 1.0, 1.0]),
                           timeout=5.0):
        # Make sure the mesh file exists
        if not path.exists(mesh_path):
            self.get_logger().error(f"File '{mesh_path}' does not exist")
            rclpy.shutdown()
            exit(1)
        # Determine ID of the collision mesh
        object_id = path.basename(mesh_path).split(".")[0]   
        
        # Add collision mesh
        self.get_logger().info(
            f"Adding collision mesh '{filepath}' "
            f"{{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}}}"
        )

        # Load the mesh if specified
        mesh = None
        mesh = trimesh.load(filepath)
        filepath = None
        if not isinstance(mesh, trimesh.Trimesh):
            self.get_logger().error(f"Failed to load mesh from '{mesh_path}'")
            rclpy.shutdown()
            exit(1)
            
        self._moveit2.add_collision_mesh(
            filepath=filepath,
            id=object_id,
            position=position,
            quat_xyzw=quat_xyzw,
            scale=scale,
            mesh=mesh,
        )
        self.get_logger().info(f"Collision mesh '{object_id}' added successfully.")
        
    
    def add_collision_primitive(self, 
                                id,
                                primitive_type, 
                                dimensions: Tuple[float, float, float], 
                                position=np.array([0.0, 0.0, 0.0]),
                                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0])):
        self._moveit2.add_collision_primitive(
            id=id,
            primitive_type=primitive_type,
            dimensions=dimensions,
            position=position,
            quat_xyzw=quat_xyzw
        )
        pass
        
    # test
    def _print_joint_states(self):
        joint_states = self.get_joint_states()
        ee_pose = self.get_ee_pose(frame="base_link")
        
        if joint_states:
            self.get_logger().info(f"\n{pformat(joint_states)}")
        else:
            self.get_logger().warn("Waiting for joint states...")
        if ee_pose is not None:
                self.get_logger().info(f"End-effector pose in base_link frame:\n{ee_pose}")
        else:
            self.get_logger().warn("End-effector pose not available.")


# def main(args=None):
#     rclpy.init(args=args)
#     robot = FrankaPandaServer(robot_ip='10.90.90.144', gripper_type=None)
#     executor = rclpy.executors.MultiThreadedExecutor()
#     executor.add_node(robot)
    
#     try:
#         # Get current EE pose
#         timeout = 5.0  # seconds
#         start_time = time.time()
#         # while robot.get_ee_pose() is None and time.time() - start_time < timeout:
#         #     rclpy.spin_once(robot, timeout_sec=0.1) 
        
#         # H_current = robot.get_ee_pose()
#         # if H_current is None:
#         #     robot.get_logger().error("Current EE pose not available.")
#         # else:
#         #     # Offset in Z by +0.05 m (5 cm)
#         #     H_target = np.array(H_current)
#         #     H_target[2, 3] = H_target[2, 3] + 0.05

#         #     robot.get_logger().info("Planning motion with +5cm Z offset...")
#         #     robot.plan_and_move_to_pose(H_target)
            
#         #     print("Current State: " + str(robot.moveit2.query_state()))
#         # rate = robot.create_rate(10)
#         # while robot.moveit2.query_state() != MoveIt2State.EXECUTING:
#         #     rate.sleep()

#         # # Get the future
#         # print("Current State: " + str(robot.moveit2.query_state()))
#         # future = robot.moveit2.get_execution_future()

#         # # Wait until the future is done
#         # while not future.done():
#         #     rate.sleep()
#         # Wait for joint state to become available
#         timeout = 5.0
#         start = time.time()
#         while robot.get_joint_positions() is None and time.time() - start < timeout:
#             rclpy.spin_once(robot, timeout_sec=0.1)

#         # Get current joint positions as a list
#         joint_pos_dict = robot.get_joint_positions()
#         if not joint_pos_dict:
#             robot.get_logger().error("Failed to get current joint positions.")
#         else:
#             joint_names = list(joint_pos_dict["fr3_arm"].keys())
#             joint_positions = list(joint_pos_dict["fr3_arm"].values())

#             print("Current Joint Positions: ", joint_positions)
#             # Modify joint 4 (index 3)
#             joint_index = 3
#             joint_positions[joint_index] += 0.1

#             robot.get_logger().info(
#                 f"Moving from current config with joint {joint_names[joint_index]} increased by 0.1"
#             )
#             robot.plan_and_move_to_configuration(joint_positions)


#         executor.spin()
#     finally:
#         robot.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()