#!/usr/bin/env python3
from parallel_parameter_search.simulators import PybulletSim, WebotsSim
from parallel_parameter_search.walk_optimization import AbstractWalkOptimization
from dataclasses import make_dataclass
import pandas as pd
from bitbots_msgs.msg import JointCommand
import math
import numpy as np

class EvaluateWalk(AbstractWalkOptimization):

    def __init__(self, namespace, gui, robot, sim_type="webots", foot_link_names=()):
        super().__init__(robot, config_name=f"walking_{robot}_simulator")
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.node, gui, robot, world="optimization_" + robot, ros_active=False)
        else:
            print(f'sim type {sim_type} not known')
        self.robot = robot
        self.time_limit = 10
        self.repetitions = 1

        for parameter_msg in self.walk_parameters:
            if parameter_msg.name == "engine.trunk_pitch":
                self.trunk_pitch = parameter_msg.value.double_value
            if parameter_msg.name == "engine.trunk_height":
                self.trunk_height = parameter_msg.value.double_value
            if parameter_msg.name == "engine.trunk_pitch_p_coef_forward":
                self.trunk_pitch_p_coef_forward = parameter_msg.value.double_value
            if parameter_msg.name == "engine.trunk_pitch_p_coef_turn":
                self.trunk_pitch_p_coef_turn = parameter_msg.value.double_value
        if self.trunk_pitch is None or self.trunk_height is None or self.trunk_pitch_p_coef_turn is None or \
                self.trunk_pitch_p_coef_forward is None:
            print("Parameters not set correctly")
            exit()
        self.reset_height_offset = 0.12
    
    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        # TODO: This should automatically use the values set in the corresponding method
        # of a Class in walk_engine_optimisation.py.
        # Can't really do that right now, because the constructor probably initialises a sim.
        joint_command_msg.joint_names = ["LElbow", "RElbow", "LShoulderPitch", "RShoulderPitch"]
        joint_command_msg.positions = [math.radians(35.86), math.radians(-36.10), math.radians(75.27),
                                       math.radians(-75.58)]
        return joint_command_msg


    def evaluate_walk(self):
        Result = make_dataclass("Result",
                                [("v_x", float), ("v_y", float), ("v_yaw", float), ("fall", bool), ("pose_obj", float),
                                 ("x_speed_multiplier", float), ("y_speed_multiplier", float),
                                 ("yaw_speed_multiplier", float)])
        maximal_speeds = []
        results = []

        def test_speed(start_speed, increase, direction):
            speed = start_speed
            while True:
                falls = 0
                speed[0] += increase[0]
                speed[1] += increase[1]
                speed[2] += increase[2]
                for i in range(self.repetitions):
                    # fully reset pose & position, otherwise the run starts improperly
                    self.reset()
                    fall, pose_obj, orientation_obj, gyro_obj, end_poses = \
                        self.evaluate_direction(*speed, self.time_limit)
                    goal_end_pose = end_poses[0]
                    actual_end_pose = end_poses[1]

                    distance_travelled_in_correct_direction = np.dot(direction, np.array(actual_end_pose))
                    # need to use factor as we have acceleration and deccaleration phases
                    actual_speed = distance_travelled_in_correct_direction / 7.5
                    print(f"actual speed: {actual_speed}")
                    for i in range(3): 
                        if speed[i] != 0:                       
                            print(f"reached speed factor: {actual_speed/speed[i]}")
                            print(f"factor for config: {speed[i]/actual_speed}")

                    real_speed_multipliers = []
                    #if goal_end_pose[0] == 0:
                    #    real_speed_multipliers.append(1)
                    #else:
                    #    real_speed_multipliers.append(goal_end_pose[0] / actual_end_pose[0])
                    #if goal_end_pose[1] == 0:
                    #    real_speed_multipliers.append(1)
                    #else:
                    #    real_speed_multipliers.append(goal_end_pose[1] / actual_end_pose[1])
                    #if goal_end_pose[2] == 0:
                    #    real_speed_multipliers.append(1)
                    #else:
                    #    real_speed_multipliers.append(goal_end_pose[2] / actual_end_pose[2])
                    #results.append(Result(*speed, fall, pose_obj, *real_speed_multipliers))
                    falls += fall

                if falls == self.repetitions:
                    print(f"Fall at {speed}")
                    print("")
                    maximal_speeds.append(speed)
                    break

        test_speed([0.0, 0, 0], [0.05, 0, 0], [1,0,0])
        test_speed([-0.0, 0, 0], [-0.05, 0, 0], [-1,0,0])
        test_speed([0, 0.0, 0], [0, 0.025, 0], [0,1,0])
        test_speed([0, 0, 0], [0, 0, 0.25], [0,0,1])

        # viewpoints
        # forward 2.5 6.2 2.5 | 0 -0.56 -0.82 3.1443
        # backward -2.5 6.2 2.5 | 0 -0.56 -0.82 3.1443
        # sideward -3 1 0.8 | 0.5 -0.5 -0.7 1.93
        # turn -2.5 0 1 | 0.5 -0.5 -0.7 1.93
        # on field -2.3 1.5 1 | -0.28 0.56 0.77 -2.42


        # MRL 0.75 -0.8 0.25 3.75
        # Wolfgang 0.55 -0.55 0.275 1.75
        # op2 0.45 -0.45 0.2 5.0
        # op3 0.8 -1.2 0.3 4.5
        # nao 0.75 -1.05 0.55 1.5
        # rfc 0.55 -0.55 0.4 2.5
        # chape 0.6 -0.7 0.425 4.0
        # nugus 0.5 -0.7 0.375 2.5
        # bez 0.35 -0.1 0.2 3.25

        while False:
            #test_speed([0.35, 0, 0], [0, 0, 0], [1,0,0])
            #test_speed([-0.1, 0, 0], [0, 0, 0], [-1,0,0])
            #test_speed([0, 0.3, 0], [0, 0, 0], [0,1,0])
            #test_speed([0, 0, 2.25], [0, 0, 0], [0,0,1])
            pass

        print(maximal_speeds)
        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_pickle(f"./walk_evaluation_{self.robot}.pkl")


walk_evaluation = EvaluateWalk("worker", True, "wolfgang")
walk_evaluation.evaluate_walk()

walk_evaluation.sim.close()
