from ament_index_python import get_package_share_directory
from parallel_parameter_search.walk_optimization import AbstractWalkOptimization

from parallel_parameter_search.simulators import PybulletSim


class AbstractWalkStabilization(AbstractWalkOptimization):
    def __init__(self, gui, robot_name, sim_type='pybullet', repetitions=1, multi_objective=False, foot_link_names=(),
                 start_terrain_height=0.0):
        super().__init__(robot_name)
        self.start_terrain_height = start_terrain_height

        if sim_type == 'pybullet':
            urdf_path = get_package_share_directory(f"{robot_name}_description") + "/urdf/robot.urdf"
            self.sim = PybulletSim(self.node, gui, urdf_path=urdf_path, foot_link_names=foot_link_names,
                                   terrain_height=self.start_terrain_height)
        elif sim_type == 'webots':
            print("Webots currently does not support stabilization optimization")
        else:
            print(f'sim type {sim_type} not known')

        # needed to reset robot pose correctly
        self.trunk_height = self.node.get_parameter("engine.trunk_height")
        self.trunk_pitch = self.node.get_parameter("engine.trunk_pitch")
        self.trunk_pitch_p_coef_forward = self.node.get_parameter("engine.trunk_pitch_p_coef_forward")
        self.trunk_pitch_p_coef_turn = self.node.get_parameter("engine.trunk_pitch_p_coef_turn")

        self.foot_x_client = self.node.get_parameter('pid_foot_pos_x')
        self.foot_y_client = self.node.get_parameter('pid_foot_pos_y')
        self.hip_pitch_client = self.node.get_parameter('pid_hip_pitch')
        self.hip_roll_client = self.node.get_parameter('pid_hip_roll')
        self.ankle_pitch_client_left = self.node.get_parameter('pid_ankle_left_pitch')
        self.ankle_roll_client_left = self.node.get_parameter('pid_ankle_left_roll')
        self.ankle_pitch_client_right = self.node.get_parameter('pid_ankle_right_pitch')
        self.ankle_roll_client_right = self.node.get_parameter('pid_ankle_right_roll')
        self.trunk_pitch_client = self.node.get_parameter('pid_trunk_pitch')
        self.trunk_roll_client = self.node.get_parameter('pid_trunk_roll')
        self.trunk_fused_pitch_client = self.node.get_parameter('pid_trunk_fused_pitch')
        self.trunk_fused_roll_client = self.node.get_parameter('pid_trunk_fused_roll')
        self.gyro_pitch_client = self.node.get_parameter('pid_trunk_pitch')
        self.gyro_roll_client = self.node.get_parameter('pid_trunk_roll')

    def objective(self, trial):
        # get parameter to evaluate from optuna
        self.suggest_walk_params(trial)
        # reset simulation with initial terrain height
        self.sim.randomize_terrain(self.start_terrain_height)
        self.reset()
        # reset walk controller
        self.walk.reset()

        cost = 0
        for iteration in range(1, self.number_of_iterations + 1):
            d = 0
            for direction in self.directions:
                d += 1
                self.reset_position()
                early_term, cost_try, end_poses, _ = self.evaluate_direction(*direction, trial, 1, self.time_limit)
                cost += cost_try
                # check if we failed in this direction and terminate this trial early
                if early_term:
                    # terminate early and give 100 cost for each try left
                    return 1 * (self.number_of_iterations - iteration) * len(self.directions) + 1 * (
                            len(self.directions) - d) + cost
            # increase difficulty of terrain
            next_height = self.start_terrain_height + self.height_per_iteration * iteration
            self.sim.randomize_terrain(next_height)
            print(f"height {next_height}")
        return cost

    def _suggest_walk_params(self, trial, pressure_reset=False, effort_reset=False, phase_rest=False,
                             stability_stop=False, foot_pid=False, hip_pid=False, ankle_pid=False, trunk_fused=False,
                             trunk_rpy=False, gyro=False):
        # optimal engine parameters are already loaded from yaml
        node_param_dict = {}

        def add(name, min_value, max_value):
            node_param_dict[name] = trial.suggest_uniform(name, min_value, max_value)

        def fix(name, value):
            node_param_dict[name] = value
            trial.set_user_attr(name, value)

        # walk engine should update at same speed as simulation
        fix("engine_freq", 1 / self.sim.get_timestep())
        # make sure that steps are not limited
        fix("max_step_x", 100.0)
        fix("max_step_y", 100.0)
        fix("max_step_xy", 100.0)
        fix("max_step_z", 100.0)
        fix("max_step_angular", 100.0)
        fix("debug_active", True)
        # activate features depending on what is optimized
        fix("pressure_phase_reset_active", pressure_reset)
        if pressure_reset:
            add("ground_min_pressure", 0, 5)
        fix("effort_phase_reset_active", effort_reset)
        fix("phase_rest_active", phase_rest)

        fix("imu_active", stability_stop)
        if stability_stop:
            add("pause_duration", 0.0, 2.0)
            add("imu_pitch_threshold", 0.0, 1.0)
            add("imu_roll_threshold", 0.0, 1.0)
            add("imu_pitch_vel_threshold", 0.0, 10.0)
            add("imu_roll_vel_threshold", 0.0, 10.0)

        def pid_params(name, client, p, i, d, i_clamp):
            pid_dict = {"p": trial.suggest_uniform(name + "_p", p[0], p[1]),
                        "d": trial.suggest_uniform(name + "_d", i[0], i[1]),
                        "i": trial.suggest_uniform(name + "_i", d[0], d[1]),
                        "i_clamp_min": i_clamp[0],
                        "i_clamp_max": i_clamp[1]}
            if isinstance(client, list):
                for c in client:
                    self.set_params(pid_dict, client, self.walk)
            else:
                self.set_params(pid_dict, client, self.walk)

        if foot_pid:
            pid_params("foot_x", self.foot_x_client, (-0.1, 0.1), (-0.1, 0.1), (-0.0, 0.0), (-10, 10))
            pid_params("foot_y", self.foot_y_client, (-0.1, 0.1), (-0.1, 0.1), (-0.0, 0.0), (-10, 10))

        if hip_pid:
            pid_params("hip_pitch", self.hip_pitch_client, (-2, 2), (-0.2, 0.2), (-0.001, 0.001), (-1, 1))
            pid_params("hip_roll", self.hip_roll_client, (-2, 2), (-0.2, 0.2), (-0.001, 0.001), (-1, 1))

        if ankle_pid:
            pid_params("ankle_pitch", [self.ankle_pitch_client_left, self.ankle_pitch_client_right], (-10, 10), (-1, 1),
                       (-0.1, 0.1), (-10, 10))
            pid_params("ankle_roll", [self.ankle_roll_client_left, self.ankle_roll_client_right], (-10, 10), (-1, 1),
                       (-0.1, 0.1), (-10, 10))

        if trunk_fused:
            pid_params("fused_pitch", self.trunk_fused_pitch_client, (-2, 2), (-0.2, 0.2), (-0.001, 0.001), (-1, 1))
            pid_params("fused_roll", self.trunk_fused_roll_client, (-2, 2), (-0.2, 0.2), (-0.001, 0.001), (-1, 1))

        if trunk_rpy:
            pid_params("trunk_pitch", self.trunk_pitch_client, (-2, 2), (-0.2, 0.2), (-0.001, 0.001), (-1, 1))
            pid_params("trunk_roll", self.trunk_roll_client, (-2, 2), (-0.2, 0.2), (-0.001, 0.001), (-1, 1))

        if gyro:
            pid_params("gyro_pitch", self.gyro_pitch_client, (-2, 2), (-0.2, 0.2), (-0.001, 0.001), (-1, 1))
            pid_params("gyro_roll", self.gyro_roll_client, (-2, 2), (-0.2, 0.2), (-0.001, 0.001), (-1, 1))

        if self.walk_as_node:
            self.set_params(node_param_dict)
        else:
            self.current_params = node_param_dict
            self.walk.set_node_dyn_reconf(node_param_dict)


class WolfgangWalkStabilization(AbstractWalkStabilization):
    def __init__(self, gui, sim_type='pybullet'):
        super(WolfgangWalkStabilization, self).__init__(gui, "wolfgang", sim_type, repetitions=1, multi_objective=False,
                                                        start_terrain_height=0.0)
        self.reset_height_offset = 0.005
        self.height_per_iteration = 0.0025
        start_speeds = (0.4, 0.2, 0.25)
        self.directions = [[start_speeds[0], 0, 0],
                           # [0, start_speeds[1], 0],
                           # [0, 0, start_speeds[2]],
                           # [-start_speeds[0], - start_speeds[1], 0],
                           # [-start_speeds[0], 0, start_speeds[2]],
                           # [start_speeds[0], start_speeds[1], start_speeds[2]]
                           ]

    def suggest_walk_params(self, trial):
        self._suggest_walk_params(trial, phase_rest=True, pressure_reset=True, trunk_rpy=True)
