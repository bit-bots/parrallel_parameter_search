import optuna

from parallel_parameter_search.simulators import WebotsSim
import rclpy
import os
from rclpy.node import Node
import random
from bitbots_utils.utils import load_moveit_parameter, get_parameters_from_ros_yaml

import tf_transformations
from bitbots_msgs.msg import JointCommand
from bitbots_quintic_walk_py.py_walk import PyWalk
from ament_index_python import get_package_share_directory
from rcl_interfaces.msg import Parameter, ParameterValue
import math
from geometry_msgs.msg import Twist
import time
# define class with objective function
class CaptureStepOptimization:
    def __init__(self, gui, robot_name):
        self.current_speed = None
        self.last_time = 0
        self.number_of_iterations = 100
        self.time_limit = 10
        self.robot_name = robot_name
        # need to init ROS for python and c++ code
        rclpy.init()
        # initialize node with unique name to avoid clashes when running the same optimization multiple times
        # needs to start with letter
        self.namespace = "anon_" + str(os.getpid()) + "_" + str(random.randint(0, 10000000)) + "_"
        self.node: Node = Node(self.namespace + "optimizer", allow_undeclared_parameters=True)
        self.sim = WebotsSim(self.node, gui, robot_name, world="optimization_" + robot_name, ros_active=False)


        # load moveit config values
        moveit_parameters = load_moveit_parameter(self.robot_name)

        # load walk params
        self.walk_parameters = get_parameters_from_ros_yaml("walking",
                                                       f"{get_package_share_directory('bitbots_quintic_walk')}"
                                                       f"/config/walking_wolfgang_simulator.yaml",
                                                       use_wildcard=True)
        # activate IK reset only for wolfgang
        self.walk_parameters.append(Parameter(name="node.ik_reset", value=ParameterValue(bool_value=self.robot_name == "wolfgang")))
        self.reset_rpy_offset = [0, 0, 0]
        self.reset_height_offset = 0.02
        self.trunk_height = 0.43
        self.trunk_pitch = 0.105566178884548#self.walk_parameters["engine.trunk_pitch"]

        # create walk as python class to call it later
        self.walk = PyWalk(self.namespace, self.walk_parameters, moveit_parameters)

        time.sleep(10)
        self.reset()


    def objective(self, trial : optuna.Trial):
        print(f"starting trial {trial.number}")
        p_pitch = trial.suggest_float("p_pitch", 0.0, 10.0)
        d_pitch = trial.suggest_float("d_pitch", 0.0, 10.0)
        threshold_pitch = trial.suggest_float("threshold_pitch", 0.0, 10.0)
        
        ## TODO fix this for the pid contorller ros parameters
        parameters = {}
        parameters["node.step_length.pitch.p"] = p_pitch
        parameters["node.step_length.pitch.d"] = d_pitch
        parameters["node.step_length.pitch.threshold"] = threshold_pitch

        self.walk.set_parameters(parameters)

        self.reset()
        fallen = False
        while not fallen:
            # step a little
            self.set_cmd_vel(x=0.1, y=0.0, yaw=0.0)
            self.complete_walking_step(number_steps=1)
            # push
            # ...
            # check fallen...
        
        return 0.0 # value of the objective function


    def has_robot_fallen(self):
        pos, rpy = self.sim.get_robot_pose_rpy()
        return abs(rpy[0]) > math.radians(45) or abs(rpy[1]) > math.radians(45) or pos[2] < self.trunk_height / 2

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

    def reset_position(self):
        height = self.trunk_height + self.reset_height_offset
        pitch = self.trunk_pitch
        (x, y, z, w) = tf_transformations.quaternion_from_euler(self.reset_rpy_offset[0],
                                                                self.reset_rpy_offset[1] + pitch,
                                                                self.reset_rpy_offset[2])

        self.sim.reset_robot_pose((0, 0, height), (x, y, z, w))

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["LElbow", "RElbow", "LShoulderPitch", "RShoulderPitch"]
        joint_command_msg.positions = [math.radians(35.86), math.radians(-36.10), math.radians(75.27),
                                       math.radians(-75.58)]
        return joint_command_msg

    def complete_walking_step(self, number_steps=1, fake=False):
        start_time = self.sim.get_time()
        for i in range(number_steps):
            # does a double step
            while rclpy.ok():
                current_time = self.sim.get_time()
                dt = current_time - self.last_time
                joint_command = self.walk.step(dt, self.current_speed,
                                               self.sim.get_imu_msg(),
                                               self.sim.get_joint_state_msg(), self.sim.get_pressure_left(),
                                               self.sim.get_pressure_right())
                self.sim.set_joints(joint_command)
                self.last_time = current_time
                if not fake:
                    self.sim.step_sim()
                phase = self.walk.get_phase()
                self.walk.spin_ros()

                # phase will advance by the simulation time times walk frequency
                next_phase = phase + self.sim.get_timestep() * self.walk.get_freq()
                # do double step to always have torso at same position
                if (phase >= 0.5 and next_phase >= 1.0):
                    # next time the walking step will change
                    break
                if self.sim.get_time() - start_time > 5:
                    # if walking does not perform step correctly due to impossible IK problems, do a timeout
                    break


optimization = CaptureStepOptimization(True, "wolfgang")

study = optuna.create_study(direction="maximize")

study.optimize(optimization.objective, n_trials=10)

print(study.best_trial)
