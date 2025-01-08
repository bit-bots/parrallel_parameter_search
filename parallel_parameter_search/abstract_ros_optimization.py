import rclpy
from bitbots_moveit_bindings.libbitbots_moveit_bindings import initRos
from rclpy.node import Node
import random
import os


class AbstractRosOptimization:

    def __init__(self, robot_name, wandb=False):
        self.robot_name = robot_name
        self.wandb = wandb
        # need to init ROS for python and c++ code
        rclpy.init()
        # initialize node with unique name to avoid clashes when running the same optimization multiple times
        # needs to start with letter
        self.namespace = "anon_" + str(os.getpid()) + "_" + str(random.randint(0, 10000000)) + "_"
        self.node: Node = Node(self.namespace + "optimizer", allow_undeclared_parameters=True)
        # make all nodes use simulation time via /clock topic
        # actually not since we use direct python interfaces and the simulation runs in the same thread
        # self.node.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.current_params = None
        self.sim = None

    def objective(self, trial):
        """
        The actual optimization target for optuna.
        """
        raise NotImplementedError()
