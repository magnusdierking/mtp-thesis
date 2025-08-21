#!/usr/bin/env python3
# ROS 2 Jazzy + MoveItPy version of your FR3Robot
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.callback_groups import ReentrantCallbackGroup

from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

# --- MoveItPy imports ---
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils.substitutions import Xacro


def get_moveit_configs():
    # Packages that actually contain the files
    # Adjust these if your package names differ
    desc_pkg = "franka_description"            # URDF/Xacro usually lives here
    cfg_pkg  = "franka_fr3_moveit_config"      # your MoveIt config package

    desc_share = get_package_share_directory(desc_pkg)
    cfg_share  = get_package_share_directory(cfg_pkg)

    # Likely locations — adjust to what you found in step (1)
    urdf_xacro = os.path.join(desc_share, "robots", "fr3", "fr3.urdf.xacro")
    # Try to use an SRDF .xacro if available; else a plain .srdf
    srdf_xacro = os.path.join(desc_share, "robots", "fr3", "fr3.srdf.xacro")
    srdf_plain = os.path.join(desc_share, "robots", "fr3", "fr3.srdf")

    # Choose which SRDF path exists
    if os.path.exists(srdf_xacro):
        srdf_subst = Xacro(srdf_xacro)
    elif os.path.exists(srdf_plain):
        srdf_subst = srdf_plain
    else:
        raise FileNotFoundError(
            f"Could not find SRDF in {cfg_share}/config (looked for fr3.srdf(.xacro)). "
            "Add the SRDF or point to the correct file."
        )

    # If your URDF is parameterized, pass xacro args here (examples below)
    #urdf_subst = Xacro(urdf_xacro)  # e.g., Xacro(urdf_xacro, hand:="false", world_frame:="fr3_link0")

    cfg = (
        MoveItConfigsBuilder(robot_name="fr3", package_name=cfg_pkg)
        .robot_description(file_path=urdf_xacro)
        .robot_description_semantic(file_path=srdf_xacro)
        # Optionally stitch in the YAMLs your package DOES have:
        .planning_pipelines(
            pipelines={"ompl": os.path.join(cfg_share, "config", "ompl_planning.yaml")},
            default_planning_pipeline="ompl",
        )
        .trajectory_execution(os.path.join(cfg_share, "config", "fr3_ros_controllers.yaml"))
        .joint_limits(os.path.join(cfg_share, "config", "fr3_joint_limits.yaml"))
        .robot_description_kinematics(os.path.join(cfg_share, "config", "kinematics.yaml"))
        # controllers yaml naming varies; include if used by your launch
    )

    return cfg.to_moveit_configs().to_dict()


class FR3Robot(Node):
    """
    FR3 control using MoveItPy (official MoveIt Python API).
    - Plan/execute to a 4x4 pose
    - Plan/execute to a joint configuration
    - Subscribes to your existing state topics
    """

    def __init__(
        self,
        node_name: str = "fr3_robot",
        group_name: str = "fr3_arm",
        base_frame: str = "fr3_link0",   # <-- change if your base frame differs
        ee_link: str = "fr3_link8",      # <-- change if your EE link differs
    ):
        super().__init__(node_name)

        # Parallel callback group (like before)
        self._cbg = ReentrantCallbackGroup()

        # --- MoveItPy setup ---
        # Create the MoveItPy "robot" and get a PlanningComponent for the arm group
        cfg_dict = get_moveit_configs()
        if 'moveit_cpp' in cfg_dict:
            self.get_logger().info(
                f"moveit_cpp: { {k: cfg_dict['moveit_cpp'][k] for k in cfg_dict['moveit_cpp'] if k in ('planning_pipelines','default_planning_pipeline')} }"
            )
        cfg_dict.setdefault('moveit_cpp', {})
        cfg_dict['moveit_cpp']['planning_pipelines'] = ['ompl']           # list of names
        cfg_dict['moveit_cpp']['default_planning_pipeline'] = 'ompl'      # the chosen one

                
        self.robot = MoveItPy(node_name="fr3", 
                              config_dict=cfg_dict,)
        self.arm = self.robot.get_planning_component(group_name)
        
        self.joint_names = ["fr3_joint1", 
                            "fr3_joint2", 
                            "fr3_joint3", 
                            "fr3_joint4", 
                            "fr3_joint5", 
                            "fr3_joint6", 
                            "fr3_joint7"]
        self.group_name = group_name
        self.base_frame = base_frame
        self.ee_link = ee_link

        # self.arm = self.robot.get_planning_component(self.group_name)

        # # Get joint names from the SRDF so we don't hardcode them
        # robot_model = self.robot.get_robot_model()
        # jmg = robot_model.get_joint_model_group(self.group_name)
        # if jmg is None:
        #     raise RuntimeError(
        #         f"Joint model group '{self.group_name}' not found. "
        #         "Make sure your MoveIt config matches this group name."
        #     )
        # self.joint_names = list(jmg.get_variable_names())

        # # Optional: set scaling factors (silently ignore if not supported)
        # try:
        #     self.arm.set_max_velocity_scaling_factor(1.0)
        #     self.arm.set_max_acceleration_scaling_factor(1.0)
        # except Exception as e:
        #     self.get_logger().warn(f"Could not set scaling factors: {e}")

        # A simple "home" configuration (adjust to your FR3 layout if needed)
        # Provide values for *your* group joint order
        self._home_configuration = {
            name: val
            for name, val in zip(
                self.joint_names,
                [
                    0.0,
                    -np.pi / 4,
                    0.0,
                    -3 * np.pi / 4,
                    0.0,
                    np.pi / 2,
                    np.pi / 4,
                ],
            )
        }

        # --- State I/O like your original ---
        self._current_pose_stamped = None
        self.create_subscription(
            PoseStamped,
            "/franka_robot_state_broadcaster/current_pose",
            self._ee_pose_callback,
            QoSProfile(depth=5),
            callback_group=self._cbg,
        )

        self._current_joint_state = None
        self.create_subscription(
            JointState,
            "/franka_robot_state_broadcaster/measured_joint_states",
            self._joint_state_callback,
            QoSProfile(depth=5),
            callback_group=self._cbg,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        

        # Periodic printout
        self.create_timer(1.0, self._timer_callback)

        self.get_logger().info(
            f"MoveItPy ready | group='{self.group_name}', base='{self.base_frame}', ee='{self.ee_link}'"
        )
        self.get_logger().info(f"Joints ({len(self.joint_names)}): {self.joint_names}")

    # ---------------------------
    # Timers / Callbacks / Helpers
    # ---------------------------

    
    def _timer_callback(self):
        # Joint state
        joint_state = self._extract_current_joint_state()
        if joint_state:
            self.get_logger().info("=== Joint State ===")
            for name, values in joint_state.items():
                pos = f"{values['position']: .3f}" if values['position'] is not None else "   None"
                vel = f"{values['velocity']: .3f}" if values['velocity'] is not None else "   None"
                eff = f"{values['effort']: .3f}" if values['effort'] is not None else "   None"
                self.get_logger().info(f"  {name:15s} pos={pos:>8}, vel={vel:>8}, eff={eff:>8}")
        else:
            self.get_logger().warn("Joint state not available.")

        # EE pose
        pose = self._extract_current_pose()
        if pose:
            pos, quat = pose
            self.get_logger().info("=== End-Effector Pose ===")
            self.get_logger().info(
                f"  Position: x={pos[0]: .3f}, y={pos[1]: .3f}, z={pos[2]: .3f}"
            )
            self.get_logger().info(
                f"  Orientation (xyzw): "
                f"x={quat[0]: .3f}, y={quat[1]: .3f}, z={quat[2]: .3f}, w={quat[3]: .3f}"
            )
        else:
            self.get_logger().warn("Pose not available.")

        # Transform
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform("optitrack", "objectPushT", now)
            t = trans.transform.translation
            r = trans.transform.rotation
            self.get_logger().info("=== Transform optitrack → objectPushT ===")
            self.get_logger().info(
                f"  Translation: x={t.x: .3f}, y={t.y: .3f}, z={t.z: .3f}"
            )
            self.get_logger().info(
                f"  Rotation (xyzw): x={r.x: .3f}, y={r.y: .3f}, z={r.z: .3f}, w={r.w: .3f}"
            )
        except TransformException as ex:
            self.get_logger().warn(f"Could not transform: {ex}")

    

    def _ee_pose_callback(self, msg: PoseStamped):
        self._current_pose_stamped = msg
        self.get_logger().debug(f"EE pose received in {msg.header.frame_id}")

    def _joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg
        self.get_logger().debug("JointState received")

    def _extract_current_pose(self):
        if self._current_pose_stamped is None:
            self.get_logger().warn("Current pose not available.")
            return None
        pos = np.array(
            [
                self._current_pose_stamped.pose.position.x,
                self._current_pose_stamped.pose.position.y,
                self._current_pose_stamped.pose.position.z,
            ]
        )
        quat_xyzw = np.array(
            [
                self._current_pose_stamped.pose.orientation.x,
                self._current_pose_stamped.pose.orientation.y,
                self._current_pose_stamped.pose.orientation.z,
                self._current_pose_stamped.pose.orientation.w,
            ]
        )
        return pos, quat_xyzw

    def _extract_current_joint_state(self):
        msg = self._current_joint_state
        if msg is None:
            self.get_logger().warn("Current joint state not available.")
            return None
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        data = {}
        for jn in self.joint_names:
            i = name_to_idx.get(jn, None)
            if i is None:
                data[jn] = {"position": None, "velocity": None, "effort": None}
            else:
                data[jn] = {
                    "position": msg.position[i] if i < len(msg.position) else None,
                    "velocity": msg.velocity[i] if i < len(msg.velocity) else None,
                    "effort": msg.effort[i] if i < len(msg.effort) else None,
                }
        return data

    # ---------------------------
    # MoveItPy planning helpers
    # ---------------------------

    # def _plan_and_execute(self):
    #     """Plans with current PlanningComponent goal and executes on success."""
    #     result = self.arm.plan()
    #     if result and result.trajectory:
    #         self.get_logger().info("Executing planned trajectory…")
    #         # Empty 'controllers' lets MoveIt pick appropriate controllers
    #         self.robot.execute(result.trajectory, controllers=[])
    #         return True
    #     self.get_logger().error("Planning failed (no trajectory).")
    #     return False

    # # ---------------------------
    # # Public API (like your original)
    # # ---------------------------

    # def plan_and_move_to_pose(self, target_pose: np.ndarray):
    #     """
    #     Plan and move the robot to a specified pose given as a 4x4 homogeneous transform.
    #     - target_pose: 4x4, bottom row [0 0 0 1], proper rotation (det=+1).
    #     """
    #     if (
    #         target_pose.shape != (4, 4)
    #         or not np.allclose(target_pose[3, :], [0, 0, 0, 1])
    #         or not np.isclose(np.linalg.det(target_pose[:3, :3]), 1.0, atol=1e-3)
    #     ):
    #         raise ValueError(
    #             "Pose must be a valid 4x4 homogeneous transform with a proper rotation."
    #         )

    #     pos = target_pose[:3, 3]
    #     quat_xyzw = R.from_matrix(target_pose[:3, :3]).as_quat()

    #     # Compose a PoseStamped in the robot's base frame (adjust if different)
    #     goal = PoseStamped()
    #     goal.header.frame_id = self.base_frame
    #     goal.pose.position.x = float(pos[0])
    #     goal.pose.position.y = float(pos[1])
    #     goal.pose.position.z = float(pos[2])
    #     goal.pose.orientation.x = float(quat_xyzw[0])
    #     goal.pose.orientation.y = float(quat_xyzw[1])
    #     goal.pose.orientation.z = float(quat_xyzw[2])
    #     goal.pose.orientation.w = float(quat_xyzw[3])

    #     self.get_logger().info(
    #         f"Planning to pose in {self.base_frame} -> pos {pos.tolist()}, quat(xyzw) {quat_xyzw.tolist()}"
    #     )

    #     # Start from current measured state
    #     self.arm.set_start_state_to_current_state()
    #     # Pose goal for the specified EE link
    #     self.arm.set_goal_state(pose_stamped_msg=goal, pose_link=self.ee_link)

    #     ok = self._plan_and_execute()
    #     if ok:
    #         self.get_logger().info("Move to pose successful.")
    #     else:
    #         self.get_logger().error("Failed to move to pose.")

    # def plan_and_move_to_configuration(self, joint_positions):
    #     """
    #     Plan and move to explicit joint positions (list/array in this group's joint order).
    #     """
    #     if len(joint_positions) != len(self.joint_names):
    #         raise ValueError(
    #             f"Expected {len(self.joint_names)} joint positions, got {len(joint_positions)}."
    #         )
    #     if not all(isinstance(p, (int, float, np.floating, np.integer)) for p in joint_positions):
    #         raise ValueError("All joint positions must be numeric.")

    #     # Map names -> values using the group's joint order
    #     conf = {name: float(val) for name, val in zip(self.joint_names, joint_positions)}
    #     self.get_logger().info(f"Planning to configuration: {conf}")

    #     self.arm.set_start_state_to_current_state()
    #     self.arm.set_goal_state(configuration=conf)

    #     ok = self._plan_and_execute()
    #     if ok:
    #         self.get_logger().info("Move to configuration successful.")
    #     else:
    #         self.get_logger().error("Failed to move to configuration.")

    # def move_to_home(self):
    #     """Moves the robot to a 'home' configuration defined in __init__."""
    #     # Ensure order matches self.joint_names
    #     joint_positions = [self._home_configuration[n] for n in self.joint_names]
    #     self.get_logger().info(f"Moving to home configuration: {joint_positions}")
    #     self.plan_and_move_to_configuration(joint_positions)


def main():
    rclpy.init()
    node = FR3Robot(
        # Change these if your MoveIt config uses other names
        group_name="fr3_arm",
        base_frame="fr3_link0",
        ee_link="fr3_link8",
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
