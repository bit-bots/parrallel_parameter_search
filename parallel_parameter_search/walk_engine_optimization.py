import math
import time

import numpy as np
import wandb
from ament_index_python import get_package_share_directory
from bitbots_msgs.msg import JointCommand

from parallel_parameter_search.walk_optimization import AbstractWalkOptimization

from parallel_parameter_search.simulators import PybulletSim, WebotsSim


class AbstractWalkEngine(AbstractWalkOptimization):
    def __init__(self, gui, robot_name, sim_type='pybullet', foot_link_names=(), start_speeds=None,
                 repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(robot_name, wandb=wandb)
        if sim_type == 'pybullet':
            urdf_path = get_package_share_directory(f"{robot_name}_description") + "/urdf/robot.urdf"
            self.sim = PybulletSim(self.node, gui, urdf_path=urdf_path, foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.node, gui, robot_name, world="optimization_" + robot_name, ros_active=False)
        else:
            print(f'sim type {sim_type} not known')

        if not start_speeds:
            print("please set start speeds")
            exit(1)
        self.start_speeds = start_speeds
        self.directions = [np.array([1, 0, 0]),
                           np.array([-1, 0, 0]),
                           np.array([0, 1, 0]),
                           np.array([0, 0, 1])]
        if only_forward:
            self.directions = [np.array([1, 0, 0])]
        self.repetitions = repetitions
        self.multi_objective = multi_objective

    def objective(self, trial):
        # log trial number, for wandb. this can be used to plot trial number over time and see if
        # time per trial increases
        if self.wandb:
            wandb.log({"trial_number": trial.number}, step=trial.number)
        start_time = time.time()
        # get parameter to evaluate from optuna
        self.suggest_walk_params(trial)
        self.reset()

        max_speeds = [0] * len(self.directions)
        max_wrong_speeds = [0] * len(self.directions)
        # standing as first test, is not in loop as it will only be done once
        fallen, pose_obj, orientation_obj, gyro_obj, end_poses = self.evaluate_direction(0, 0, 0, 1, standing=True)
        if fallen:
            print("not standing")
            trial.set_user_attr('termination_reason', "not standing")
            # give lower score as 0 for each direction as we did not even stand
            # or not since this confuses TPE
            #max_speeds = [-1] * len(self.directions)
        else:
            # iterate over the directions
            d = 0
            for direction in self.directions:
                # compute an array showing the wrong directions in which the robot should not move
                wrong_direction = np.array([0, 0, 0])
                for i in range(len(direction)):
                    # every direction which is zero is a wrong direction
                    if direction[i] == 0:
                        wrong_direction[i] = 1
                # constantly increase the speed
                for iteration in range(1, self.number_of_iterations + 1):
                    cmd_vel = direction * np.array(self.start_speeds) * iteration
                    fallen = False
                    mean_speed = 0
                    mean_wrong_speed = 0
                    # do multiple repetitions of the same values since behavior is not always exactly deterministic
                    for i in range(self.repetitions):
                        self.reset_position()
                        fall, pose_obj, orientation_obj, gyro_obj, (goal_pose, end_pose) = \
                            self.evaluate_direction(*cmd_vel, self.time_limit)
                        if fall:
                            fallen = True
                            break

                        # get the relevant part of the end pose with this cmd_vel
                        distance_travelled_in_correct_direction = np.dot(direction, np.array(end_pose))
                        mean_speed += distance_travelled_in_correct_direction / self.time_limit
                        distance_travelled_in_wrong_direction = np.linalg.norm(wrong_direction * np.array(end_pose))
                        mean_wrong_speed += distance_travelled_in_wrong_direction / self.time_limit

                    if fallen:
                        trial.set_user_attr('termination_reason', f"falling at {cmd_vel}")
                        print("break fall")
                        self.reset()
                        break

                    mean_speed /= self.repetitions
                    mean_wrong_speed /= self.repetitions
                    print(f"mean speed {mean_speed}")
                    print(f"mean wrong speed {mean_wrong_speed}")

                    #if mean_wrong_speed - 0.01 > mean_speed:
                    #    # we move more into the wrong direction than into the correct one.
                    #    # These parameters are not good, but we need to check this manually.
                    #    # The speed in correct direction might still increase slightly and the robot may not fall
                    #    # substract a bit wrong speed as it will otherwise be true to often for small velocities
                    #    trial.set_user_attr('termination_reason', f"movement in wrong direction at {cmd_vel}")
                    #    print("break wrong direction")
                    #    break

                    if mean_speed < max_speeds[d]:
                        # we did not manage to go further
                        trial.set_user_attr('termination_reason', f"no speed increase at {cmd_vel}")
                        print("break speed")
                        break
                    else:
                        max_speeds[d] = mean_speed
                    if mean_wrong_speed > max_wrong_speeds[d]:
                        max_wrong_speeds[d] = mean_wrong_speed
                d += 1
        if self.wandb:
            # log wall time of the trial
            wandb.log({"trial_wall_duration": time.time() - start_time}, step=trial.number)

        if self.multi_objective:
            return max_speeds  # + max_wrong_speeds
        else:
            if len(self.directions) == 4:
                # use different weighting factors for the different directions
                return max_speeds[0] + max_speeds[1] + 2 * max_speeds[2] + 0.2 * max_speeds[3]
                # - max_wrong_speeds[0] - max_wrong_speeds[1] - max_wrong_speeds[2] - max_wrong_speeds[3]
            elif len(self.directions) == 1:
                return max_speeds[0]
            else:
                print("scalarization not implemented")

    def _suggest_walk_params(self, trial, trunk_height, foot_distance, foot_rise, trunk_x, z_movement):
        param_dict = {}

        def add(name, min_value, max_value):
            param_dict[name] = float(trial.suggest_float(name, min_value, max_value))

        def fix(name, value):
            param_dict[name] = float(value)
            trial.set_user_attr(name, value)

        add('engine.double_support_ratio', 0.0, 0.5)
        add('engine.freq', 1, 2)

        add('engine.foot_distance', foot_distance[0], foot_distance[1])
        add('engine.trunk_height', trunk_height[0], trunk_height[1])

        add('engine.trunk_phase', -0.5, 0.5)
        add('engine.trunk_swing', 0.0, 1.0)
        add('engine.trunk_z_movement', 0, z_movement)

        add('engine.trunk_x_offset', -trunk_x, trunk_x)
        add('engine.trunk_y_offset', -trunk_x, trunk_x)

        add('engine.trunk_pitch_p_coef_forward', -1, 1)
        add('engine.trunk_pitch_p_coef_turn', -0.5, 0.5)

        add('engine.trunk_pitch', -0.5, 0.5)
        # fix('trunk_pitch', 0.0)
        add('engine.foot_rise', foot_rise[0], foot_rise[1])
        # fix('foot_rise', foot_rise)

        add('engine.foot_pitch_angle', 3.0, 15.0)

        # add('engine.first_step_swing_factor', 0.0, 2)
        fix('engine.first_step_swing_factor', 1.80591386587488)
        fix('engine.first_step_trunk_phase', -0.5)

        # add('foot_overshoot_phase', 0.0, 1.0)
        # add('foot_overshoot_ratio', 0.0, 1.0)
        fix('engine.foot_overshoot_phase', 1.0)
        fix('engine.foot_overshoot_ratio', 0.0)

        fix('engine.foot_apex_phase', 0.5)

        # add('foot_z_pause', 0, 1)
        # add('foot_put_down_phase', 0, 1)
        # add('trunk_pause', 0, 1)
        fix('engine.foot_z_pause', 0.0)
        fix('engine.foot_put_down_phase', 1.0)
        fix('engine.trunk_pause', 0.0)

        fix('engine.foot_put_down_z_offset', 0.0)
        fix('engine.trunk_x_offset_p_coef_forward', 0.0)
        fix('engine.trunk_x_offset_p_coef_turn', 0.0)

        # walk engine should update at same speed as simulation
        param_dict["node.engine_freq"] = 1 / self.sim.get_timestep()
        # don't use loop closure when optimizing parameter
        #param_dict["node.pressure_phase_reset_active"] = False
        #param_dict["node.effort_phase_reset_active"] = False
        # make sure that steps are not limited
        # param_dict["node.imu_active"] = False
        param_dict["node.max_step_x"] = 100.0
        param_dict["node.max_step_y"] = 100.0
        param_dict["node.max_step_xy"] = 100.0
        param_dict["node.max_step_z"] = 100.0
        param_dict["node.max_step_angular"] = 100.0
        param_dict["node.x_speed_multiplier"] = 1.0
        param_dict["node.y_speed_multiplier"] = 1.0
        param_dict["node.yaw_speed_multiplier"] = 1.0

        self.current_params = param_dict
        self.walk.set_parameters(param_dict)

        # necessary for correct reset
        self.trunk_height = self.current_params["engine.trunk_height"]
        self.trunk_pitch = self.current_params["engine.trunk_pitch"]
        self.trunk_pitch_p_coef_forward = self.current_params.get("engine.trunk_pitch_p_coef_forward", 0)
        self.trunk_pitch_p_coef_turn = self.current_params.get("engine.trunk_pitch_p_coef_turn", 0)


class WolfgangWalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='pybullet', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'wolfgang', sim_type, start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                         multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.012

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.37, 0.41), foot_distance=(0.15, 0.25), foot_rise=(0.05, 0.10),
                                  trunk_x=0.02, z_movement=0.05)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["LElbow", "RElbow", "LShoulderPitch", "RShoulderPitch"]
        joint_command_msg.positions = [math.radians(35.86), math.radians(-36.10), math.radians(75.27),
                                       math.radians(-75.58)]
        return joint_command_msg


class OP2WalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='webots', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'robotis_op2', sim_type, foot_link_names=['l_sole', 'r_sole'],
                         start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                         multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.09

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.17, 0.24), foot_distance=(0.08, 0.12), foot_rise=(0.05, 0.15),
                                  trunk_x=0.03, z_movement=0.03)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["LElbow", "RElbow", "LShoulderPitch", "RShoulderPitch"]
        joint_command_msg.positions = [math.radians(-60), math.radians(60), math.radians(120.0),
                                       math.radians(-120.0)]
        return joint_command_msg


class OP3WalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='webots', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'op3', sim_type, foot_link_names=['r_ank_roll_link', 'l_ank_roll_link'],
                         start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                         multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.01

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.15, 0.25), foot_distance=(0.08, 0.12), foot_rise=(0.05, 0.15),
                                  trunk_x=0.03, z_movement=0.03)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["l_el", "r_el", "l_sho_pitch", "r_sho_pitch", "l_sho_roll", "r_sho_roll"]
        joint_command_msg.positions = [math.radians(-140), math.radians(140), math.radians(-135),
                                       math.radians(135), math.radians(-90), math.radians(90)]
        return joint_command_msg


class NaoWalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='webots', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'nao', sim_type, foot_link_names=['l_ankle', 'r_ankle'], start_speeds=[0.05, 0.025, 0.25],
                         repetitions=repetitions, multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.01

        if sim_type == 'pybullet':
            self.sim.set_joints_dict(
                {"LShoulderPitch": 1.57, "RShoulderPitch": 1.57, 'LShoulderRoll': 0.3, 'RShoulderRoll': -0.3})

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.28, 0.34), foot_distance=(0.08, 0.16), foot_rise=(0.05, 0.1),
                                  trunk_x=0.05, z_movement=0.025)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["LShoulderPitch", "RShoulderPitch", 'LShoulderRoll', 'RShoulderRoll']
        joint_command_msg.positions = [1.57, 1.57, 0.3, -0.3]
        return joint_command_msg


class RFCWalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='pybullet', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'rfc', sim_type, start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                         multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.011

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.3, 0.4), foot_distance=(0.15, 0.22), foot_rise=(0.05, 0.15),
                                  trunk_x=0.05, z_movement=0.05)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["LeftElbow", "RightElbow", "LeftShoulderPitch",
                                         "RightShoulderPitch"]
        joint_command_msg.positions = [math.radians(-90.0), math.radians(90.0), math.radians(45.0),
                                       math.radians(-45.0)]
        return joint_command_msg


class ChapeWalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='webots', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'chape', sim_type, foot_link_names=['l_sole', 'r_sole'],
                         start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                         multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.15

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.22, 0.27), foot_distance=(0.08, 0.12), foot_rise=(0.01, 0.15),
                                  trunk_x=0.03, z_movement=0.03)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["leftElbowYaw", "rightElbowYaw", "leftShoulderPitch",
                                         "rightShoulderPitch", "leftShoulderYaw", "rightShoulderYaw"]
        joint_command_msg.positions = [math.radians(-160), math.radians(160), math.radians(75.27),
                                       math.radians(75.58), math.radians(-75.58), math.radians(75.58)]
        return joint_command_msg


class MRLHSLWalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='pybullet', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'mrl_hsl', sim_type, start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                         multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.24

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.25, 0.32), foot_distance=(0.15, 0.22), foot_rise=(0.05, 0.15),
                                  trunk_x=0.05, z_movement=0.05)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["Shoulder-L", "Shoulder-R", "UpperArm-L",
                                         "UpperArm-R", "LowerArm-L", "LowerArm-R"]
        joint_command_msg.positions = [math.radians(60.0), math.radians(-60.0), math.radians(10.0),
                                       math.radians(-10.0), math.radians(-135.0), math.radians(135.0)]
        return joint_command_msg


class NugusWalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='pybullet', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'nugus', sim_type, start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                         multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.012

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.4, 0.5), foot_distance=(0.15, 0.25), foot_rise=(0.05, 0.15),
                                  trunk_x=0.05, z_movement=0.05)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["left_elbow_pitch", "right_elbow_pitch", "left_shoulder_pitch",
                                         "right_shoulder_pitch", "left_shoulder_roll", "right_shoulder_roll"]
        joint_command_msg.positions = [math.radians(-120), math.radians(-120), math.radians(120),
                                       math.radians(120), math.radians(20), math.radians(-20)]
        return joint_command_msg


class SAHRV74WalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='pybullet', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'sahrv74', sim_type, start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                         multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.01

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.3, 0.38), foot_distance=(0.15, 0.22), foot_rise=(0.05, 0.15),
                                  trunk_x=0.05, z_movement=0.05)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["left_shoulder_pitch", "right_shoulder_pitch",
                                         "left_shoulder_roll",
                                         "right_shoulder_roll", "left_elbow", "right_elbow"]
        joint_command_msg.positions = [math.radians(60.0), math.radians(60.0), math.radians(10.0),
                                       math.radians(10.0), math.radians(-135.0), math.radians(-135.0)]
        return joint_command_msg


class BezWalkEngine(AbstractWalkEngine):
    def __init__(self, gui, sim_type='webots', repetitions=1, multi_objective=False, only_forward=False, wandb=False):
        super().__init__(gui, 'bez', sim_type, foot_link_names=['l_sole', 'r_sole'],
                         start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                         multi_objective=multi_objective, only_forward=only_forward, wandb=wandb)
        self.reset_height_offset = 0.15

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, trunk_height=(0.14, 0.2), foot_distance=(0.07, 0.14), foot_rise=(0.03, 0.08),
                                  trunk_x=0.03, z_movement=0.05)

    def get_arm_pose(self):
        joint_command_msg = JointCommand()
        joint_command_msg.joint_names = ["left_arm_motor_0",
                                         "right_arm_motor_0", "left_arm_motor_1", "right_arm_motor_1"]
        joint_command_msg.positions = [math.radians(0),
                                       math.radians(0), math.radians(170), math.radians(170)]
        return joint_command_msg
