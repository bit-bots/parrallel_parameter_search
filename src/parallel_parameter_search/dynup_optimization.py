#!/usr/bin/env python3
import time

import dynamic_reconfigure.client
from bitbots_msgs.msg import DynUpActionGoal, DynUpActionResult, JointCommand

import math

import roslaunch
import rospkg
import rospy
import tf

from parallel_parameter_search.abstract_ros_optimization import AbstractRosOptimization
from parallel_parameter_search.utils import set_param_to_file, load_yaml_to_param
from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from sensor_msgs.msg import Imu


class AbstractDynupOptimization(AbstractRosOptimization):
    def __init__(self, namespace, gui, robot, abort_threshold, sim_type, foot_link_names=()):
        super().__init__(namespace)
        self.rospack = rospkg.RosPack()
        # set robot urdf and srdf
        load_robot_param(self.namespace, self.rospack, robot)
        self.sim_type = sim_type
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui, robot)
        else:
            print(f'sim type {sim_type} not known')
        # load dynup params
        load_yaml_to_param(self.namespace, 'bitbots_dynup',
                           '/config/dynup_sim.yaml',
                           self.rospack)

        self.dynup_node = roslaunch.core.Node('bitbots_dynup', 'DynupNode', 'dynup',
                                              namespace=self.namespace)
        self.robot_state_publisher = roslaunch.core.Node('robot_state_publisher', 'robot_state_publisher',
                                                         'robot_state_publisher',
                                                         namespace=self.namespace)
        self.dynup_node.remap_args = [("/tf", "tf"), ("dynup_motor_goals", "DynamixelController/command"),
                                      ("/tf_static", "tf_static"), ("/clock", "clock")]
        self.robot_state_publisher.remap_args = [("/tf", "tf"), ("/tf_static", "tf_static"), ("/clock", "clock")]
        load_yaml_to_param("/robot_description_kinematics", robot + '_moveit_config',
                           '/config/kinematics.yaml', self.rospack)
        self.launch.launch(self.robot_state_publisher)
        self.launch.launch(self.dynup_node)

        self.dynup_request_pub = rospy.Publisher(self.namespace + '/dynup/goal', DynUpActionGoal, queue_size=10)
        self.dynamixel_controller_pub = rospy.Publisher(self.namespace + "/DynamixelController/command", JointCommand)
        self.number_of_iterations = 10
        self.time_limit = 20
        self.time_difference = 0
        self.reset_height_offset = None
        self.reset_rpy_offset = [0, math.pi / 2, 0]
        self.trunk_height = 0.38  # rosparam.get_param(self.namespace + "/dynup/trunk_height")
        self.trunk_pitch = 0.0

        self.result_subscriber = rospy.Subscriber(self.namespace + "/dynup/result", DynUpActionResult, self.result_cb)
        self.dynup_complete = False

        self.imu_subscriber = rospy.Subscriber(self.namespace + "/imu/data", Imu, self.imu_cb)

        self.rise_phase_time = 0
        self.total_trial_length = 0

        self.imu_offset_sum = 0
        self.trunk_y_offset_sum = 0
        self.trial_failed_loss = 0
        self.trial_duration = 0
        self.trial_running = False
        self.dynup_params = {}
        self.abort_threshold = abort_threshold

        self.dynup_client = dynamic_reconfigure.client.Client(self.namespace + '/' + 'dynup/', timeout=60)

    def result_cb(self, msg):
        self.dynup_complete = True
        rospy.logerr("Dynup complete.")

    def imu_cb(self, msg):
        pos, rpy = self.sim.get_robot_pose_rpy()

        if self.trial_duration > self.rise_phase_time:  # only account for pitch in the last phase
            self.imu_offset_sum += abs(rpy[1])
        if not 1.56 < rpy[1] < 1.59:  # make sure to ignore states with gimbal lock
            self.imu_offset_sum += abs(rpy[2])
            self.imu_offset_sum += abs(rpy[0])
        self.trunk_y_offset_sum += abs(pos[1])

    def objective(self, trial):
        # for testing transforms
        while False:
            self.sim.set_robot_pose_rpy([0, 0, 1], [0.0, 0.0, 0.4])
            self.sim.step_sim()
            pos, rpy = self.sim.get_robot_pose_rpy()
            print(f"x: {round(pos[0], 2)}")
            print(f"y: {round(pos[1], 2)}")
            print(f"z: {round(pos[2], 2)}")
            print(f"roll: {round(rpy[0], 2)}")
            print(f"pitch: {round(rpy[1], 2)}")
            print(f"yaw: {round(rpy[2], 2)}")
            time.sleep(1)

        self.suggest_params(trial)

        self.dynup_params = rospy.get_param(self.namespace + "/dynup")
        self.reset()
        self.run_attempt()
        return self.imu_offset_sum + self.trunk_y_offset_sum + 200 * self.trial_failed_loss

    def run_attempt(self):
        self.trial_running = True
        msg = DynUpActionGoal()
        msg.goal.direction = "front"
        self.dynup_request_pub.publish(msg)
        start_time = self.sim.get_time()
        second_time = self.sim.get_time() + self.time_limit  # todo better name for second_time
        while not rospy.is_shutdown():
            self.sim.step_sim()
            print(self.sim.get_link_pose("head")[2])
            if self.sim.get_time() - start_time > self.abort_threshold and self.sim.get_link_pose("head")[2] < 0.15:  # early abort
                self.trial_failed_loss = self.total_trial_length - (self.sim.get_time() - start_time)
                return
            if self.dynup_complete:
                second_time = self.sim.get_time()
                self.dynup_complete = False
            if self.sim.get_time() - start_time > self.time_limit or self.sim.get_time() - second_time > 3:
                self.trial_duration = self.sim.get_time() - start_time
                self.dynup_complete = False
                self.trial_running = False
                return
        # self.time_difference = self.sim.get_time() - start_time

    def reset_position(self):
        height = self.trunk_height + self.reset_height_offset
        pitch = self.trunk_pitch

        if self.sim_type == "pybullet":
            (x, y, z, w) = tf.transformations.quaternion_from_euler(self.reset_rpy_offset[0],
                                                                    self.reset_rpy_offset[1] + pitch,
                                                                    self.reset_rpy_offset[2])

            self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))
        else:
            angle = math.pi / 2
            x = 0
            y = 1
            z = 0
            self.sim.reset_robot_pose((0, 0, height), (angle, x, y, z))

    def reset(self):
        # reset params
        self.imu_offset_sum = 0
        self.trunk_y_offset_sum = 0
        self.trial_failed_loss = 0

        try:
            self.rise_phase_time = self.dynup_params["time_hands_side"] + \
                                   self.dynup_params["time_hands_rotate"] + \
                                   self.dynup_params["time_foot_close"] + \
                                   self.dynup_params["time_hands_front"] + \
                                   self.dynup_params["time_foot_ground_front"] + \
                                   self.dynup_params["time_torso_45"] + \
                                   self.dynup_params["time_to_squat"]
            self.total_trial_length = self.rise_phase_time + \
                                      self.dynup_params["wait_in_squat_front"] + \
                                      self.dynup_params["rise_time"] + 3
            # trial continues for 3 sec after dynup completes
        except KeyError:
            rospy.logwarn("Parameter server not available yet, continuing anyway.")
            self.rise_phase_time = 100  # todo: hacky high value that should never be reached. But it works
        # reset simulation
        self.sim.set_gravity(False)
        self.sim.reset_robot_pose((0, 0, 1), (0, 0, 0, 1))
        time = self.sim.get_time()

        while self.sim.get_time() - time < 2:
            msg = JointCommand()
            msg.joint_names = ["HeadPan", "HeadTilt", "LElbow", "LShoulderPitch", "LShoulderRoll", "RElbow",
                               "RShoulderPitch", "RShoulderRoll", "LHipYaw", "LHipRoll", "LHipPitch", "LKnee",
                               "LAnklePitch", "LAnkleRoll", "RHipYaw", "RHipRoll", "RHipPitch", "RKnee", "RAnklePitch",
                               "RAnkleRoll"]
            # msg.positions = [0, 0, 0.79, 0, 0, -0.79, 0, 0, -0.01, 0.06, 0.47, 1.01, -0.45, 0.06, 0.01, -0.06, -0.47,
            #                 -1.01, 0.45, -0.06] #walkready
            msg.positions = [0, 0.78, 0.78, 1.36, 0, -0.78, -1.36, 0, 0.11, 0.07, -0.19, 0.23, -0.63, 0.07, 0.11, -0.07,
                             0.19, -0.23, 0.63, -0.07]  # falling_front
            self.dynamixel_controller_pub.publish(msg)
            self.sim.step_sim()
        self.reset_position()
        self.sim.set_gravity(True)
        time = self.sim.get_time()
        while not self.sim.get_time() - time > 2:
            self.sim.step_sim()


class WolfgangOptimization(AbstractDynupOptimization):
    def __init__(self, namespace, gui, abort_threshold, sim_type='pybullet'):
        super(WolfgangOptimization, self).__init__(namespace, gui, 'wolfgang', abort_threshold, sim_type)
        self.reset_height_offset = 0.005

    def suggest_params(self, trial):
        node_param_dict = {}

        def add(name, min_value, max_value):
            node_param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            node_param_dict[name] = value
            trial.set_user_attr(name, value)

        add("max_leg_angle", 20, 80)
        add("foot_distance", 0.106, 0.3)
        add("leg_min_length", 0.18, 0.25)
        add("arm_side_offset", 0.05, 0.2)
        add("trunk_x", -0.2, 0.2)

        self.set_params(node_param_dict, self.dynup_client)
        return


def load_robot_param(namespace, rospack, name):
    rospy.set_param(namespace + '/robot_type_name', name)
    set_param_to_file(namespace + "/robot_description", name + '_description', '/urdf/robot.urdf', rospack)
    set_param_to_file(namespace + "/robot_description_semantic", name + '_moveit_config',
                      '/config/' + name + '.srdf', rospack)
    load_yaml_to_param(namespace + "/robot_description_kinematics", name + '_moveit_config',
                       '/config/kinematics.yaml', rospack)
    load_yaml_to_param(namespace + "/robot_description_planning", name + '_moveit_config',
                       '/config/joint_limits.yaml', rospack)