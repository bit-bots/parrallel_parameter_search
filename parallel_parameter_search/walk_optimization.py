from ament_index_python import get_package_share_directory

from bitbots_msgs.msg import JointCommand
from bitbots_quintic_walk_py.py_walk import PyWalk

import math
import numpy as np
import optuna
import rclpy
from geometry_msgs.msg import Twist
import tf_transformations
from rcl_interfaces.msg import Parameter, ParameterValue

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.simulators import WebotsSim

from bitbots_utils.utils import load_moveit_parameter, get_parameters_from_ros_yaml


class AbstractWalkOptimization(AbstractRosOptimization):

    def __init__(self, robot_name, wandb=False, config_name="walking_wolfgang_simulator"):
        super().__init__(robot_name, wandb=wandb)
        self.current_speed = None
        self.last_time = 0
        self.number_of_iterations = 100
        self.time_limit = 10
        # needs to be specified by subclasses
        self.directions = None
        self.reset_height_offset = None
        self.reset_rpy_offset = [0, 0, 0]
        self.last_yaw = 0
        self.summed_yaw = 0

        # load moveit config values
        moveit_parameters = load_moveit_parameter(self.robot_name)

        # load walk params
        self.walk_parameters = get_parameters_from_ros_yaml("walking",
                                                       f"{get_package_share_directory('bitbots_quintic_walk')}"
                                                       f"/config/{config_name}.yaml",
                                                       use_wildcard=True)
        # activate IK reset only for wolfgang
        self.walk_parameters.append(Parameter(name="node.ik.reset", value=ParameterValue(bool_value=self.robot_name == "wolfgang")))

        # create walk as python class to call it later
        self.walk = PyWalk(self.namespace, self.walk_parameters + moveit_parameters)

    def suggest_walk_params(self, trial):
        raise NotImplementedError

    def correct_pitch(self, x, y, yaw):
        return self.trunk_pitch + self.trunk_pitch_p_coef_forward * x + self.trunk_pitch_p_coef_turn * yaw

    def has_robot_fallen(self):
        pos, rpy = self.sim.get_robot_pose_rpy()
        return abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.trunk_height / 2

    def modulate_speed(self, x, y, yaw, passed_time, time_limit, standing):
        # change speed first increasing and then decreasing
        if passed_time < 1:
            self.set_cmd_vel(x / 4.0, y / 4.0, yaw / 4.0, stop=standing)
        elif passed_time < 2:
            self.set_cmd_vel(x / 2.0, y / 2.0, yaw / 2.0, stop=standing)
        elif passed_time > time_limit:
            # reached time limit, stop robot
            self.set_cmd_vel(0, 0, 0, stop=True)
        elif passed_time > time_limit - 1:
            # decelerate
            self.set_cmd_vel(x / 4.0, y / 4.0, yaw / 4.0, stop=standing)
        elif passed_time > time_limit - 2:
            # decelerate
            self.set_cmd_vel(x / 2.0, y / 2.0, yaw / 2.0, stop=standing)
        else:
            # set normal speed in the middle
            self.set_cmd_vel(x, y, yaw, stop=standing)

    def track_pose(self, pos, rpy):
        # we need to sum the yaw manually to recognize complete turns
        current_yaw = rpy[2]
        if self.last_yaw > math.tau / 4 and current_yaw < 0:
            # counterclockwise turn
            self.last_yaw = -math.tau / 2 - (math.tau / 2 - self.last_yaw)
        elif self.last_yaw < -math.tau / 4 and current_yaw > 0:
            # clockwise turn
            self.last_yaw = math.tau / 2 + (math.tau / 2 + self.last_yaw)
        self.summed_yaw += current_yaw - self.last_yaw
        self.last_yaw = current_yaw
        return [pos[0], pos[1], self.summed_yaw]

    def evaluate_direction(self, x, y, yaw, time_limit, standing=False):
        if time_limit == 0:
            raise AssertionError("Time limit must be greater than 0")  # todo when is this happening?
        print(F'cmd: {round(x, 2)} {round(y, 2)} {round(yaw, 2)}')
        start_time = self.sim.get_time()
        orientation_diff = 0.0
        angular_vel_diff = 0.0
        self.last_yaw = 0
        self.summed_yaw = 0

        # set up dict for to record joint states of each time step
        joint_recording = {
            'name': self.sim.get_joint_state_msg().name,
            'position_series': [],
            'velocity_series': [],
            'effort_series': [],
            'walk_phase': [],
            'time_sec_series': [],
            'time_nano_series': [],
            'sim_time_series': [],
            'command_start_time': start_time,
            'command_sec_series': [],
            'command_x_series': [],
        }

        # wait till time for test is up or stopping condition has been reached
        while rclpy.ok():
            # track time
            passed_time = self.sim.get_time() - start_time
            passed_timesteps = passed_time / self.sim.get_timestep()
            if passed_timesteps == 0:
                # edge case with division by zero
                passed_timesteps = 1

            # manage speed for slow increase and decrease at start and end
            self.modulate_speed(x, y, yaw, passed_time, time_limit, standing)

            # track pose to count full turns of the robot
            pos, rpy = self.sim.get_robot_pose_rpy()
            current_pose = self.track_pose(pos, rpy)

            # get orientation diff scaled to 0-1
            orientation_diff += min(1, (abs(rpy[0]) + abs(rpy[1] - self.correct_pitch(x, y, yaw))) * 0.5)
            imu_msg = self.sim.get_imu_msg()
            # get angular_vel diff scaled to 0-1. dont take yaw, since we might actually turn around it
            angular_vel_diff += min(1, (abs(imu_msg.angular_velocity.x) + abs(imu_msg.angular_velocity.y)) / 60)

            if passed_time > time_limit + 2:
                # robot should have stopped now, evaluate the fitness
                pose_cost, poses = self.compute_cost(x, y, yaw, current_pose)
                return False, pose_cost, orientation_diff / passed_timesteps, angular_vel_diff / passed_timesteps, poses, joint_recording

            # test if the robot has fallen down
            if self.has_robot_fallen():
                pose_cost, poses = self.compute_cost(x, y, yaw, current_pose)
                return True, pose_cost, orientation_diff / passed_timesteps, angular_vel_diff / passed_timesteps, poses, joint_recording

            # set commands to simulation and step
            current_time = self.sim.get_time()
            joint_command = self.walk.step(current_time - self.last_time, self.current_speed,
                                           imu_msg,
                                           self.sim.get_joint_state_msg(),
                                           self.sim.get_pressure_left(), self.sim.get_pressure_right())
            self.sim.set_joints(joint_command)
            self.last_time = current_time
            self.sim.step_sim()

            # self.walk.publish_debug()
            # spine py+cpp nodes just to allow introspection from terminal and create debug messages if necessary
            # there is no spin_some method in python, just try to do it a couple of times
            # TODO this could be done in a better way if rclpy had a spin_some method
            for i in range(5):
                rclpy.spin_once(self.node, timeout_sec=0)
            self.walk.spin_ros()

            # record joint state. This is done after stepping the sim with the current commands,
            # to get a clean correlation of commands and effects on the same time-step.

            joint_state = self.sim.get_joint_state_msg()
            joint_recording['position_series'].append(joint_state.position)
            joint_recording['velocity_series'].append(joint_state.velocity)
            joint_recording['effort_series'].append(joint_state.effort)

            joint_recording['walk_phase'].append(self.walk.get_phase())

            nanos = joint_state.header.stamp.sec * 1e9 + joint_state.header.stamp.nanosec
            joint_recording['time_nano_series'].append(nanos)
            joint_recording['time_sec_series'].append(nanos * 1e-9)
            joint_recording['sim_time_series'].append(self.sim.get_time())
            joint_recording['command_sec_series'].append(passed_time)
            joint_recording['command_x_series'].append(self.current_speed.linear.x)

        # was stopped before finishing
        raise optuna.exceptions.OptunaError()

    def complete_walking_step(self, number_steps=1, fake=False):
        start_time = self.sim.get_time()
        for i in range(number_steps):
            # does a double step
            while rclpy.ok():
                current_time = self.sim.get_time()
                joint_command = self.walk.step(current_time - self.last_time, self.current_speed,
                                               self.sim.get_imu_msg(),
                                               self.sim.get_joint_state_msg(), self.sim.get_pressure_left(),
                                               self.sim.get_pressure_right())
                self.sim.set_joints(joint_command)
                self.last_time = current_time
                if not fake:
                    self.sim.step_sim()
                phase = self.walk.get_phase()
                # phase will advance by the simulation time times walk frequency
                next_phase = phase + self.sim.get_timestep() * self.walk.get_freq()
                # do double step to always have torso at same position
                if (phase >= 0.5 and next_phase >= 1.0):
                    # next time the walking step will change
                    break
                if self.sim.get_time() - start_time > 5:
                    # if walking does not perform step correctly due to impossible IK problems, do a timeout
                    break

    def compute_cost(self, v_x, v_y, v_yaw, current_pose):
        def get_matrix(v_x, v_y, v_yaw, t):
            if v_yaw == 0:
                # prevent division by zero
                return (np.array([v_x * t, v_y * t]), 0)
            else:
                return (np.array([(v_x * math.sin(t * v_yaw) - v_y * (1 - math.cos(t * v_yaw))) / v_yaw,
                                  (v_y * math.sin(t * v_yaw) + v_x * (1 - math.cos(t * v_yaw))) / v_yaw]), v_yaw * t)

        def rotation(yaw):
            return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

        # compute end pose. robot is basically walking on circles where radius=v_x/v_yaw and v_y/v_yaw
        # we need to include the acceleration and deceleration phases
        # 1s first accel, 1s second accel, time_limit -3 full speed, 1s deceleration
        # always need to rotate around current yaw
        after_first_accel = get_matrix(v_x / 4, v_y / 4, v_yaw / 4, 1)
        after_second_accel = get_matrix(v_x / 2, v_y / 2, v_yaw / 2, 1)
        after_second_accel = (after_first_accel[0] + np.dot(rotation(after_first_accel[1]), after_second_accel[0]),
                              after_first_accel[1] + after_second_accel[1])
        after_full_speed = get_matrix(v_x, v_y, v_yaw, self.time_limit - 3)
        after_full_speed = (after_second_accel[0] + np.dot(rotation(after_second_accel[1]), after_full_speed[0]),
                            after_second_accel[1] + after_full_speed[1])
        after_deceleration = get_matrix(v_x / 2, v_y / 2, v_yaw / 2, 1)
        after_deceleration = (after_full_speed[0] + np.dot(rotation(after_full_speed[1]), after_deceleration[0]),
                              after_full_speed[1] + after_deceleration[1])

        # back to x,y,yaw format
        correct_pose = (after_deceleration[0][0], after_deceleration[0][1], after_deceleration[1])

        # we need to handle targets cleverly or we will have divisions by 0
        if v_x == 0 and v_y == 0:
            trans_target = 1
        else:
            # Pythagoras
            trans_target = math.sqrt(correct_pose[0] ** 2 + correct_pose[1] ** 2)
        # always take tau as measurement
        rot_target = math.tau

        # Pythagoras
        trans_error_abs = math.sqrt((correct_pose[0] - current_pose[0]) ** 2 + (correct_pose[1] - current_pose[1]) ** 2)
        # yaw is split in continuous sin and cos components
        rot_error_abs = abs(math.sin(correct_pose[2]) - math.sin(current_pose[2])) + abs(
            math.cos(correct_pose[2]) - math.cos(current_pose[2]))
        pose_cost = ((trans_error_abs / trans_target) + (rot_error_abs / rot_target)) / 2

        # scale to [0-1]
        if pose_cost > 1:
            print("cutting pose cost to normalize it to [0-1]")
        pose_cost = min(1, pose_cost)

        return pose_cost, (correct_pose, current_pose)

    def reset_position(self):
        height = self.trunk_height + self.reset_height_offset
        pitch = self.trunk_pitch
        (x, y, z, w) = tf_transformations.quaternion_from_euler(self.reset_rpy_offset[0],
                                                                self.reset_rpy_offset[1] + pitch,
                                                                self.reset_rpy_offset[2])

        self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))

    def reset(self):
        # reset simulation
        # let the robot do a few steps in the air to get correct walkready position
        self.sim.set_gravity(False)
        self.sim.set_self_collision(False)
        if isinstance(self.sim, WebotsSim):
            # fix for strange webots physic errors
            self.sim.reset_robot_init()
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1), reset_joints=True)
        # set arms correctly
        joint_command_msg = self.get_arm_pose()
        self.sim.set_joints(joint_command_msg)
        self.set_cmd_vel(0.1, 0, 0)
        self.complete_walking_step()
        self.set_cmd_vel(0, 0, 0, stop=True)
        self.complete_walking_step()
        self.sim.set_gravity(True)
        # some robots have issues in their collision model
        if self.robot_name not in ["nao", "chape"]:
            self.sim.set_self_collision(True)
        self.reset_position()

    def set_cmd_vel(self, x: float, y: float, yaw: float, stop=False):
        msg = Twist()
        msg.linear.x = float(x)
        msg.linear.y = float(y)
        msg.linear.z = 0.0
        msg.angular.z = float(yaw)
        if stop:
            msg.angular.x = -1.0
        self.current_speed = msg
