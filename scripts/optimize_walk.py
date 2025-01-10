#!/usr/bin/env python3

import argparse
import importlib
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler, NSGAIISampler # TODO MOTPESampler,
from optuna.integration import WeightsAndBiasesCallback
import numpy as np

from parallel_parameter_search.walk_engine_optimization import OP2WalkEngine, WolfgangWalkEngine, OP3WalkEngine, \
    NaoWalkEngine, RFCWalkEngine, ChapeWalkEngine, MRLHSLWalkEngine, NugusWalkEngine, SAHRV74WalkEngine, BezWalkEngine, SigmabanWalkEngine

from parallel_parameter_search.walk_stabilization import WolfgangWalkStabilization

parser = argparse.ArgumentParser()
parser.add_argument('--storage', help='Database SQLAlchemy string, e.g. postgresql://USER:PASS@SERVER/DB_NAME',
                    default=None, type=str, required=False)
parser.add_argument('--name', help='Name of the study', default=None, type=str, required=True)
parser.add_argument('--robot', help='Robot model that should be used {wolfgang, op2, op3, nao, rfc, chape, mrl_hsl, nugus, bez, sahrv74} ',
                    default=None, type=str, required=True)
parser.add_argument('--sim', help='Simulator type that should be used {pybullet, webots} ', default=None, type=str,
                    required=True)
parser.add_argument('--gui', help="Activate gui", action='store_true')
parser.add_argument('--type', help='Optimization type that should be used {engine, stabilization} ', default=None,
                    type=str, required=True)
parser.add_argument('--startup', help='Startup trials', default=-1,
                    type=int, required=False)
parser.add_argument('--trials', help='Trials to be evaluated', default=10000,
                    type=int, required=True)
parser.add_argument('--sampler', help='Which sampler {MOTPE, TPE, CMAES, NSGA2, Random}', default=10000,
                    type=str, required=True)
parser.add_argument('--repetitions', help='How often each trial is repeated while beeing evaluated', default=1,
                    type=int, required=False)
parser.add_argument('--suggest', help='Suggest a working solution', action='store_true')
parser.add_argument('--wandb', help='Use wandb', action='store_true')
parser.add_argument('--forward', help='Only optimize forward direction', action='store_true')
parser.add_argument('--multivariate', help='Activate multivariate feature of TPE', action='store_true')
parser.add_argument('--render-video', help='Render video of best solution', action='store_true')
args = parser.parse_args()

seed = np.random.randint(2 ** 32 - 1)
n_startup_trials = args.startup

multi_objective = args.sampler in ['MOTPE', 'Random']

if args.type == "engine":
    kwargs = {"gui" : args.gui, 
              "sim_type" : args.sim,
              "repetitions" : args.repetitions,
              "multi_objective" : multi_objective,
              "only_forward" : args.forward,
              "wandb" : args.wandb, 
              "render_video" : args.render_video}
    walk_engines = {"op2" : OP2WalkEngine,
                    "wolfgang" : WolfgangWalkEngine,
                    "op3" : OP3WalkEngine,
                    "nao" : NaoWalkEngine,
                    "rfc" : RFCWalkEngine,
                    "chape" : ChapeWalkEngine,
                    "mrl_hsl" : MRLHSLWalkEngine,
                    "nugus" : NugusWalkEngine,
                    "sahrv74" : SAHRV74WalkEngine,
                    "bez" : BezWalkEngine,
                    "sigmaban" : SigmabanWalkEngine,
                    }
    if args.robot in walk_engines:
        objective = walk_engines[args.robot](**kwargs)
    else:
        print(f"robot type \"{args.robot}\" not known.")
        exit()
elif args.type == "stabilization":
    if args.robot == "wolfgang":
        objective = WolfgangWalkStabilization(gui=args.gui, sim_type=args.sim,
                                              repetitions=args.repetitions, multi_objective=multi_objective)
        # add one trial without stabilitation at the beginning to provide a baseline
        study.enqueue_trial(
            {"pitch.p": 0.0, "pitch.i": 0.0, "pitch.d": 0.0, "pitch.i_clamp_min": 0.0, "pitch.i_clamp_max": 0.0,
             "roll.p": 0.0, "roll.i": 0.0, "roll.d": 0.0, "roll.i_clamp_min": 0.0, "roll.i_clamp_max": 0.0,
             "pause_duration": 0.0, "imu_pitch_threshold": 0.0, "imu_roll_threshold": 0.0,
             "imu_pitch_vel_threshold": 0.0, "imu_roll_vel_threshold": 0.0})
    else:
        print(f"robot type \"{args.robot}\" not known.")
else:
    print(f"Optimization type {args.type} not known.")

num_variables = len(objective.directions)

if args.sampler == "TPE":
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=args.multivariate, constant_liar=True)
elif args.sampler == "CMAES":
    sampler = CmaEsSampler(n_startup_trials=n_startup_trials, seed=seed)
elif args.sampler == "MOTPE":
    if n_startup_trials == -1:
        n_startup_trials = num_variables * 11 - 1
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed, multivariate=args.multivariate, constant_liar=True)
elif args.sampler == "NSGA2":
    sampler = NSGAIISampler(seed=seed)
elif args.sampler == "Random":
    sampler = RandomSampler(seed=seed)
else:
    print("sampler not correctly specified. Should be one of {TPE, CMAES, MOTPE, NSGA2, Random}")
    exit(1)

if multi_objective:
    study = optuna.create_study(study_name=args.name, storage=args.storage,
                                directions=["maximize"] * num_variables,  # + ["minimize"] * num_variables,
                                sampler=sampler, load_if_exists=True)
else:
    study = optuna.create_study(study_name=args.name, storage=args.storage, direction="maximize",
                                sampler=sampler, load_if_exists=True)

study.set_user_attr("sampler", args.sampler)
study.set_user_attr("multivariate", args.multivariate)
study.set_user_attr("sim", args.sim)
study.set_user_attr("robot", args.robot)
study.set_user_attr("type", args.type)
study.set_user_attr("repetitions", args.repetitions)


if args.suggest:
    if args.type == "engine":
        if len(study.get_trials()) == 0:
            # old params
            print("#############\nUSING GIVEN PARAMETERS\n#############")
            for i in range(100):
                study.enqueue_trial({
                    "engine.double_support_ratio": 0.0264282002140171,
                    "engine.first_step_swing_factor": 1.80591386587488,
                    "engine.foot_distance": 0.179900277671633,
                    "engine.foot_rise": 0.0819786291304007,
                    "engine.freq": 1.2,
                    "engine.trunk_height": 0.394780002666927,
                    "engine.trunk_phase": -0.151653984431689,
                    "engine.trunk_pitch": 0.105566178884548,
                    "engine.trunk_pitch_p_coef_forward": -0.186068274875133,
                    "engine.trunk_pitch_p_coef_turn": -0.457339940581988,
                    "engine.trunk_swing": 0.154856652745882,
                    "engine.trunk_x_offset": 0.00885762700992359,
                    "engine.trunk_y_offset": 0.00369752824507733,
                    "engine.trunk_z_movement": 0.0155005052923747,
                    "engine.first_step_trunk_phase": -0.5,
                    "engine.foot_apex_phase": 0.5,
                    "engine.foot_apex_pitch": 0.1,
                    "engine.foot_overshoot_phase": 1.0,
                    "engine.foot_overshoot_ratio": 0.0,
                    "engine.foot_put_down_phase": 1.0,
                    "engine.foot_z_pause": 0.0,
                    "engine.trunk_pause": 0.0
                })

    else:
        print("no suggestion specified for this type")

# only use wandb callback if name provided
if args.wandb:
    wandb_kwargs = {
        "project": f"mujoco_quintic_walk_optim",
        "entity": "bitbots",
        "tags": [args.sampler, args.robot, args.sim],
        "resume": "never",
        #"group": args.name,  # use group so that we can run multiple studies in parallel
    }
    if args.multivariate:
        wandb_kwargs["tags"].append("multivariate")

    if multi_objective:
        if args.forward:
            metric_name = ["objective.forward"]
        else:
            metric_name = ["objective.forward", "objective.backward", "objective.left", "objective.turn"]
            # "objective.error_forward", "objective.error_backward", "objective.error_left", "objective.error_turn"],
    else:
        metric_name = ["objective"]
    wandbc = WeightsAndBiasesCallback(
        metric_name=metric_name,
        wandb_kwargs=wandb_kwargs)
    callbacks = [wandbc]
else:
    callbacks = []
study.optimize(objective.objective, n_trials=args.trials, show_progress_bar=False, callbacks=callbacks)

# close simulator window
objective.sim.close()

exit(0)
