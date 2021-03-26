# THIS HAS TO BE IMPORTED FIRST!!! I don't know why
from bitbots_msgs.msg import JointCommand
from bitbots_quintic_walk import PyWalk

import math
import time

import dynamic_reconfigure.client
import optuna
import roslaunch
import rospkg
import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param, load_robot_param
from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from sensor_msgs.msg import Imu, JointState


class AbstractWalkOptimization(AbstractRosOptimization):

    def __init__(self, namespace, robot_name, walk_as_node):
        super().__init__(namespace)
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot_name)

        # load walk params
        load_yaml_to_param(self.namespace, 'bitbots_quintic_walk',
                           '/config/walking_' + robot_name + '_optimization.yaml',
                           self.rospack)

        self.walk_as_node = walk_as_node
        self.current_speed = None
        self.last_time = 0
        if self.walk_as_node:
            self.walk_node = roslaunch.core.Node('bitbots_quintic_walk', 'WalkNode', 'walking',
                                                 namespace=self.namespace)
            self.walk_node.remap_args = [("walking_motor_goals", "DynamixelController/command"), ("/clock", "clock")]
            self.launch.launch(self.walk_node)
            self.dynconf_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'walking/engine', timeout=60)
            self.cmd_vel_pub = rospy.Publisher(self.namespace + '/cmd_vel', Twist, queue_size=10)
        else:
            load_yaml_to_param("/robot_description_kinematics", robot_name + '_moveit_config',
                               '/config/kinematics.yaml', self.rospack)
            # create walk as python class to call it later
            self.walk = PyWalk(self.namespace)

        self.number_of_iterations = 10
        self.time_limit = 10

        # needs to be specified by subclasses
        self.directions = None
        self.reset_height_offset = None
        self.reset_rpy_offset = [0, 0, 0]

    def objective(self, trial):
        raise NotImplementedError

    def suggest_walk_params(self, trial):
        raise NotImplementedError

    def correct_pitch(self, x, y, yaw):
        return self.trunk_pitch + self.trunk_pitch_p_coef_forward * x + self.trunk_pitch_p_coef_turn * yaw

    def evaluate_direction(self, x, y, yaw, trial: optuna.Trial, iteration, time_limit, start_speed=True):
        if time_limit == 0:
            time_limit = 1
        if start_speed:
            # start robot slowly
            self.set_cmd_vel(x * iteration / 4, y * iteration / 4, yaw * iteration / 4)
        else:
            self.set_cmd_vel(x * iteration, y * iteration, yaw * iteration)
        print(F'cmd: {round(x * iteration, 2)} {round(y * iteration, 2)} {round(yaw * iteration, 2)}')
        start_time = self.sim.get_time()
        orientation_diff = 0.0
        # wait till time for test is up or stopping condition has been reached
        while not rospy.is_shutdown():
            passed_time = self.sim.get_time() - start_time
            passed_timesteps = passed_time / self.sim.get_timestep()
            if passed_timesteps == 0:
                # edge case with division by zero
                passed_timesteps = 1
            if start_speed:
                if passed_time > 2:
                    # use real speed
                    self.set_cmd_vel(x * iteration, y * iteration, yaw * iteration)
                elif passed_time > 1:
                    self.set_cmd_vel(x * iteration / 2, y * iteration / 2, yaw * iteration / 2)
            if passed_time > time_limit:
                # reached time limit, stop robot
                self.set_cmd_vel(0, 0, 0)

            if passed_time > time_limit + 2:
                # robot should have stopped now, evaluate the fitness
                didnt_move, pose_cost = self.compute_cost(x * iteration, y * iteration, yaw * iteration)
                return False, didnt_move, pose_cost, orientation_diff / passed_timesteps, 1 - min(1, (
                        passed_time / (time_limit + 2)))

            # test if the robot has fallen down
            pos, rpy = self.sim.get_robot_pose_rpy()
            # get orientation diff scaled to 0-1
            orientation_diff += min(1, (abs(rpy[0]) + abs(rpy[1] - self.correct_pitch(x, y, yaw))) * 0.5)
            if abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.trunk_height / 2:
                didnt_move, pose_cost = self.compute_cost(x * iteration, y * iteration, yaw * iteration)
                return True, didnt_move, pose_cost, orientation_diff / passed_timesteps, 1 - min(1, (
                        passed_time / (time_limit + 2)))

            if self.walk_as_node:
                # give time to other algorithms to compute their responses
                # use wall time, as ros time is standing still
                time.sleep(0.01)  # todo would be better to just wait till a command from walking arrived, like in dynup
            else:
                current_time = self.sim.get_time()
                joint_command = self.walk.step(current_time - self.last_time, self.current_speed,
                                               self.sim.get_imu_msg(),
                                               self.sim.get_joint_state_msg(),
                                               self.sim.get_pressure_left(), self.sim.get_pressure_right())
                self.sim.set_joints(joint_command)
                self.last_time = current_time
            self.sim.step_sim()

        # was stopped before finishing
        raise optuna.exceptions.OptunaError()

    def run_walking(self, duration):
        start_time = self.sim.get_time()
        while not rospy.is_shutdown() and (duration is None or self.sim.get_time() - start_time < duration):
            self.sim.step_sim()
            current_time = self.sim.get_time()
            joint_command = self.walk.step(current_time - self.last_time, self.current_speed, self.sim.get_imu_msg(),
                                           self.sim.get_joint_state_msg(), self.sim.get_pressure_left(),
                                           self.sim.get_pressure_right())
            self.sim.set_joints(joint_command)
            self.last_time = current_time

    def complete_walking_step(self, number_steps=1, fake=False):
        start_time = self.sim.get_time()
        for i in range(number_steps):
            # does a double step
            while not rospy.is_shutdown():
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

    def compute_cost(self, v_x, v_y, v_yaw):
        # 2D pose
        pos, rpy = self.sim.get_robot_pose_rpy()
        current_pose = [pos[0], pos[1], rpy[2]]
        t = self.time_limit
        if v_yaw == 0:
            # we are not turning, we can compute the pose easily
            correct_pose = [v_x * self.time_limit,
                            v_y * self.time_limit,
                            v_yaw * self.time_limit]
        else:
            # compute end pose. robot is basically walking on circles where radius=v_x/v_yaw and v_y/v_yaw
            correct_pose = [(v_x * math.sin(t * v_yaw) - v_y * (1 - math.cos(t * v_yaw))) / v_yaw,
                            (v_y * math.sin(t * v_yaw) + v_x * (1 - math.cos(t * v_yaw))) / v_yaw,
                            v_yaw * t]
        # todo this does not include the acceleration phase correctly
        # weighted mean squared error, yaw is split in continuous sin and cos components
        yaw_error = (math.sin(correct_pose[2]) - math.sin(current_pose[2])) ** 2 + (math.cos(correct_pose[2]) -
                                                                                    math.cos(current_pose[2])) ** 2
        # normalize pose error
        pose_cost = ((correct_pose[0] - current_pose[0]) ** 2 + (correct_pose[1] - current_pose[1]) ** 2 + yaw_error)

        # test if robot moved at all for simple case
        didnt_move = False
        didnt_move_factor = 0.25
        if v_x >= 0:
            x_correct = current_pose[0] > didnt_move_factor * correct_pose[0]
        else:
            x_correct = current_pose[0] < didnt_move_factor * correct_pose[0]
        if v_y >= 0:
            y_correct = current_pose[1] > didnt_move_factor * correct_pose[1]
        else:
            y_correct = current_pose[1] < didnt_move_factor * correct_pose[1]

        if (v_x != 0 and not x_correct) or (v_y != 0 and not y_correct):
            didnt_move = True
            print(f"x goal {correct_pose[0]} cur {current_pose[0]}")
            print(f"y goal {correct_pose[1]} cur {current_pose[1]}")
            print("didn't move")

        # scale to [0-1]
        pose_cost = min(1, pose_cost / 20)

        return didnt_move, pose_cost

    def reset_position(self):
        height = self.trunk_height + self.reset_height_offset
        pitch = self.trunk_pitch
        (x, y, z, w) = tf.transformations.quaternion_from_euler(self.reset_rpy_offset[0],
                                                                self.reset_rpy_offset[1] + pitch,
                                                                self.reset_rpy_offset[2])

        self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))

    def reset(self):
        # reset simulation
        # let the robot do a few steps in the air to get correct walkready position
        self.sim.set_gravity(False)
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        self.set_cmd_vel(0.1, 0, 0)
        # set arms correctly
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["LElbow", "RElbow"]
        joint_command_msg.positions = [math.radians(60), math.radians(-60)]
        self.sim.set_joints(joint_command_msg)
        if self.walk_as_node:
            self.sim.run_simulation(duration=2, sleep=0.01)
        else:
            self.complete_walking_step()
        self.set_cmd_vel(0, 0, 0)
        if self.walk_as_node:
            self.sim.run_simulation(duration=2, sleep=0.01)
        else:
            self.complete_walking_step()
        self.sim.set_gravity(True)
        self.reset_position()

    def set_cmd_vel(self, x, y, yaw):
        msg = Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.linear.z = 0
        msg.angular.z = yaw
        if self.walk_as_node:
            self.cmd_vel_pub.publish(msg)
        else:
            self.current_speed = msg
