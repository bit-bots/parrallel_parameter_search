import math

import rospy
from parallel_parameter_search.walk_optimization import AbstractWalkOptimization

from parallel_parameter_search.simulators import PybulletSim, WebotsSim


class AbstractWalkEngine(AbstractWalkOptimization):
    def __init__(self, namespace, gui, robot, walk_as_node, sim_type='pybullet', foot_link_names=(), start_speeds=None,
                 repetitions=1, multi_objective=False):
        super(AbstractWalkEngine, self).__init__(namespace, robot, walk_as_node)
        if sim_type == 'pybullet':
            urdf_path = self.rospack.get_path(robot + '_description') + '/urdf/robot.urdf'
            self.sim = PybulletSim(self.namespace, gui, urdf_path=urdf_path,
                                   foot_link_names=foot_link_names)
        elif sim_type == 'webots':
            self.sim = WebotsSim(self.namespace, gui, robot, world="flat_world", ros_active=True)
        else:
            print(f'sim type {sim_type} not known')

        if not start_speeds:
            print("please set start speeds")
            exit(1)
        self.directions = [[start_speeds[0], 0, 0],
                           [0, start_speeds[1], 0],
                           [0, 0, start_speeds[2]],
                           [-start_speeds[0], - start_speeds[1], 0],
                           [-start_speeds[0], 0, start_speeds[2]],
                           [start_speeds[0], start_speeds[1], start_speeds[2]]
                           ]
        self.repetitions = repetitions
        self.multi_objective = multi_objective

    def objective(self, trial):
        # get parameter to evaluate from optuna
        self.suggest_walk_params(trial)
        self.reset()

        d = 0
        it = 0
        # standing as first test, is not in loop as it will only be done once
        fall_sum, didnt_move_sum, pose_obj, stability_obj, time_obj = self.evaluate_direction(0, 0, 0, trial, 1, 1)
        if fall_sum:
            # terminate early and give 1 cost for each try left
            trial.set_user_attr('early_termination_at', (0, 0, 0))
        else:
            # iterate over the directions to increase the speed
            for iteration in range(1, self.number_of_iterations + 1):
                it += 1
                d = 0
                do_break = False
                for direction in self.directions:
                    d += 1
                    fall_rep_sum = 0
                    didnt_move_rep_sum = 0
                    pose_obj_rep_sum = 0
                    stability_obj_rep_sum = 0
                    time_obj_rep_sum = 0
                    # do multiple repetitions of the same values since behavior is not always exactly deterministic
                    for i in range(self.repetitions):
                        self.reset_position()
                        fall, didnt_move, pose_obj, stability_obj, time_obj = self.evaluate_direction(*direction, trial,
                                                                                                      iteration,
                                                                                                      self.time_limit)
                        if fall:
                            fall_rep_sum += 1
                        if didnt_move:
                            didnt_move_rep_sum += 1
                        pose_obj_rep_sum += pose_obj
                        stability_obj_rep_sum += stability_obj
                        time_obj_rep_sum += time_obj

                    # use the mean as costs for this try
                    pose_obj += pose_obj_rep_sum / self.repetitions
                    stability_obj += stability_obj_rep_sum / self.repetitions
                    time_obj += time_obj_rep_sum / self.repetitions

                    # check if we always failed in this direction and terminate this trial early
                    if fall_rep_sum == self.repetitions or didnt_move_rep_sum == self.repetitions:
                        # terminate early and give 1 cost for each try left
                        # add extra information to trial
                        trial.set_user_attr('early_termination_at',
                                            (direction[0] * iteration, direction[1] * iteration,
                                             direction[2] * iteration))
                        do_break = True
                        break
                if do_break:
                    break

        # add costs based on the the iterations left
        directions_left = (self.number_of_iterations - it) * len(self.directions) + (len(self.directions) - d)
        fall_sum += directions_left
        pose_obj += directions_left
        stability_obj += directions_left
        time_obj += directions_left

        if self.multi_objective:
            return [fall_sum, pose_obj, stability_obj, time_obj]
        else:
            return fall_sum + pose_obj + stability_obj + time_obj

    def _suggest_walk_params(self, trial, trunk_height, foot_distance, foot_rise, trunk_x, z_movement):
        engine_param_dict = {}

        def add(name, min_value, max_value):
            engine_param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            engine_param_dict[name] = value
            trial.set_user_attr(name, value)

        add('double_support_ratio', 0.0, 0.5)
        add('freq', 0.5, 5)

        add('foot_distance', foot_distance[0], foot_distance[1])
        add('trunk_height', trunk_height[0], trunk_height[1])

        add('trunk_phase', -0.5, 0.5)
        add('trunk_swing', 0.0, 1.0)
        add('trunk_z_movement', 0, z_movement)

        add('trunk_x_offset', -trunk_x, trunk_x)
        add('trunk_y_offset', -trunk_x, trunk_x)

        add('trunk_pitch_p_coef_forward', -5, 5)
        add('trunk_pitch_p_coef_turn', -5, 5)

        add('trunk_pitch', -0.5, 0.5)
        # fix('trunk_pitch', 0.0)
        add('foot_rise', 0.05, 0.15)
        # fix('foot_rise', foot_rise)

        add('first_step_swing_factor', 0.0, 2)
        # fix('first_step_swing_factor', 1)
        fix('first_step_trunk_phase', -0.5)

        # add('foot_overshoot_phase', 0.0, 1.0)
        # add('foot_overshoot_ratio', 0.0, 1.0)
        fix('foot_overshoot_phase', 1)
        fix('foot_overshoot_ratio', 0.0)

        fix('foot_apex_phase', 0.5)

        # add('foot_z_pause', 0, 1)
        # add('foot_put_down_phase', 0, 1)
        # add('trunk_pause', 0, 1)
        fix('foot_z_pause', 0)
        fix('foot_put_down_phase', 1)
        fix('trunk_pause', 0)

        node_param_dict = {}
        # walk engine should update at same speed as simulation
        node_param_dict["engine_freq"] = 1 / self.sim.get_timestep()
        # don't use loop closure when optimizing parameter
        node_param_dict["pressure_phase_reset_active"] = False
        node_param_dict["effort_phase_reset_active"] = False
        node_param_dict["phase_rest_active"] = False
        # make sure that steps are not limited
        node_param_dict["imu_active"] = False
        node_param_dict["max_step_x"] = 100.0
        node_param_dict["max_step_y"] = 100.0
        node_param_dict["max_step_xy"] = 100.0
        node_param_dict["max_step_z"] = 100.0
        node_param_dict["max_step_angular"] = 100.0

        if self.walk_as_node:
            self.set_params(engine_param_dict)
            self.set_params(node_param_dict)
        else:
            self.current_params = engine_param_dict
            self.walk.set_engine_dyn_reconf(engine_param_dict)
            self.walk.set_node_dyn_reconf(node_param_dict)

        # necessary for correct reset
        self.trunk_height = self.current_params["trunk_height"]
        self.trunk_pitch = self.current_params["trunk_pitch"]
        self.trunk_pitch_p_coef_forward = self.current_params.get("trunk_pitch_p_coef_forward", 0)
        self.trunk_pitch_p_coef_turn = self.current_params.get("trunk_pitch_p_coef_turn", 0)


class WolfgangWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet', repetitions=1, multi_objective=False):
        super(WolfgangWalkEngine, self).__init__(namespace, gui, 'wolfgang', walk_as_node, sim_type,
                                                 start_speeds=[0.2, 0.1, 0.25], repetitions=repetitions,
                                                 multi_objective=multi_objective)
        self.reset_height_offset = 0.005

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.38, 0.42), (0.15, 0.25), 0.1, 0.03, 0.03)


class DarwinWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='webots', repetitions=1, multi_objective=False):
        super(DarwinWalkEngine, self).__init__(namespace, gui, 'darwin', walk_as_node, sim_type,
                                               foot_link_names=['MP_ANKLE2_L', 'MP_ANKLE2_R'],
                                               start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                                               multi_objective=multi_objective)
        self.reset_height_offset = 0.09

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.20, 0.24), (0.08, 0.15), 0.02, 0.02, 0.02)


class OP3WalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='webots', repetitions=1, multi_objective=False):
        super(OP3WalkEngine, self).__init__(namespace, gui, 'op3', walk_as_node, sim_type,
                                            foot_link_names=['r_ank_roll_link', 'l_ank_roll_link'],
                                            start_speeds=[0.05, 0.025, 0.25], repetitions=repetitions,
                                            multi_objective=multi_objective)
        self.reset_height_offset = 0.01

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.13, 0.24), (0.08, 0.15), 0.02, 0.02, 0.02)


class NaoWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='webots', repetitions=1, multi_objective=False):
        super(NaoWalkEngine, self).__init__(namespace, gui, 'nao', walk_as_node, sim_type,
                                            foot_link_names=['l_ankle', 'r_ankle'], start_speeds=[0.05, 0.025, 0.25],
                                            repetitions=repetitions, multi_objective=multi_objective)
        self.reset_height_offset = 0.01

        if sim_type == 'pybullet':
            self.sim.set_joints_dict(
                {"LShoulderPitch": 1.57, "RShoulderPitch": 1.57, 'LShoulderRoll': 0.3, 'RShoulderRoll': -0.3})

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.27, 0.32), (0.1, 0.17), 0.03, 0.02, 0.02)


class ReemcWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet', repetitions=1, multi_objective=False):
        super(ReemcWalkEngine, self).__init__(namespace, gui, 'reemc', walk_as_node, sim_type,
                                              foot_link_names=['leg_left_6_link', 'leg_right_6_link'],
                                              start_speeds=[0.1, 0.05, 0.5], repetitions=repetitions,
                                              multi_objective=multi_objective)
        self.reset_height_offset = -0.1
        self.reset_rpy_offset = (-0.1, 0.15, -0.5)

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.5, 0.65), (0.15, 0.30), 0.1, 0.15, 0.05)


class TalosWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet', repetitions=1, multi_objective=False):
        super(TalosWalkEngine, self).__init__(namespace, gui, 'talos', walk_as_node, sim_type,
                                              foot_link_names=['leg_left_6_link', 'leg_right_6_link'],
                                              start_speeds=[0.02, 0.01, 0.01], repetitions=repetitions,
                                              multi_objective=multi_objective)
        self.reset_height_offset = -0.13
        self.reset_rpy_offset = (0, 0.15, 0)

        if sim_type == 'pybullet':
            self.sim.set_joints_dict({"arm_left_4_joint": -1.57, "arm_right_4_joint": -1.57})

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.6, 0.75), (0.15, 0.4), 0.1, 0.15, 0.08)


class AtlasWalkEngine(AbstractWalkEngine):
    def __init__(self, namespace, gui, walk_as_node, sim_type='pybullet', repetitions=1, multi_objective=False):
        super(AtlasWalkEngine, self).__init__(namespace, gui, 'atlas', walk_as_node, sim_type,
                                              foot_link_names=['l_sole', 'r_sole'],
                                              start_speeds=[0.02, 0.01, 0.01], repetitions=repetitions,
                                              multi_objective=multi_objective)
        self.reset_height_offset = 0.0
        self.reset_rpy_offset = (0, 0, 0)

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, (0.6, 0.75), (0.15, 0.4), 0.1, 0.15, 0.08)
