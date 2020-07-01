import os
import subprocess
import sys
import time

import rospy
from wolfgang_pybullet_sim.simulation import Simulation
from wolfgang_pybullet_sim.ros_interface import ROSInterface

from darwin_description.darwin_webots_controller import DarwinWebotsController


class AbstractSim:

    def __init__(self):
        pass

    def step_sim(self):
        raise NotImplementedError

    def run_simulation(self, duration, sleep):
        start_time = rospy.get_time()
        while not rospy.is_shutdown() and (duration is None or rospy.get_time() - start_time < duration):
            self.step_sim()
            time.sleep(sleep)

    def set_gravity(self, on):
        raise NotImplementedError

    def reset_robot_pose(self, pos, quat):
        raise NotImplementedError

    def get_robot_pose_rpy(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_time(self):
        raise NotImplementedError

class PybulletSim(AbstractSim):

    def __init__(self, namespace, gui):
        super(AbstractSim, self).__init__()
        self.namespace = namespace
        self.gui = gui
        self.sim: Simulation = Simulation(gui)
        self.sim_interface: ROSInterface = ROSInterface(self.sim, namespace=self.namespace + '/', node=False)

    def step_sim(self):
        self.sim.step()

    def set_gravity(self, on):
        self.sim.set_gravity(on)

    def reset_robot_pose(self, pos, quat):
        self.sim.reset_robot_pose(pos, quat)

    def get_robot_pose_rpy(self):
        return self.sim.get_robot_pose_rpy()

    def reset(self):
        self.sim.reset()

    def get_time(self):
        return self.sim.time

    def get_imu_msg(self):
        return self.sim_interface.get_imu_msg()

    def get_joint_state_msg(self):
        return self.sim_interface.get_joint_state_msg()

    def set_joints(self, joint_command_msg):
        self.sim_interface.joint_goal_cb(joint_command_msg)


class WebotsSim(AbstractSim):

    def __init__(self, namespace, gui):
        # start webots
        super().__init__()
        arguments = ["webots",
                     "--batch",
                     "/home/marc/repositories/running_robot_competition/running_robot_environment/worlds/RunningRobotEnv_optim.wbt"]
        if not gui:
            arguments.append("--minimize")
        sim_proc = subprocess.Popen(arguments)

        os.environ["WEBOTS_PID"] = str(sim_proc.pid)
        if gui:
            mode = 'run'
        else:
            mode = 'fast'
        self.robot_controller = DarwinWebotsController(namespace, False, mode)

    def step_sim(self):
        self.robot_controller.step()

    def set_gravity(self, on):
        self.robot_controller.set_gravity(on)

    def reset_robot_pose(self, pos, quat):
        self.robot_controller.reset_robot_pose(pos, quat)

    def get_robot_pose_rpy(self):
        return self.robot_controller.get_robot_pose_rpy()

    def reset(self):
        self.robot_controller.reset()

    def get_time(self):
        return self.robot_controller.time

    def get_imu_msg(self):
        return self.robot_controller.get_imu_msg()

    def get_joint_state_msg(self):
        return self.robot_controller.get_joint_state_msg()

    def set_joints(self, joint_command_msg):
        self.robot_controller.command_cb(joint_command_msg)

# todo gazebo