'''
multiple intersection, independent dqn/rule based policy
'''
import argparse
import json
import logging
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd

import cityflow
from cityflow_env import CityFlowEnvM
# from test.cityflow_env import CityFlowEnv
from utility import parse_roadnet, plot_data_lists
from dqn_agent import MDQNAgent

# import ray

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GPU


def main():
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    cityflow_config = {
        "interval": 1,
        "seed": 0,
        "laneChange": False,
        "dir": "data/",
        "roadnetFile": "roadnet_1_3.json",
        "flowFile": "anon_1_3_700_0.3_synthetic.json",
        "rlTrafficLight": False,
        "saveReplay": True,
        "roadnetLogFile": "replayRoadNet.json",
        "replayLogFile": "replayLogFile.txt"
    }

    with open(os.path.join("data/", "cityflow.config"), "w") as json_file:
        json.dump(cityflow_config, json_file)

    config = {
        'cityflow_config_file': "data/cityflow.config",
        'epoch': 20,
        'num_step': 3600,  # 每个epoch的执行步数
        'save_freq': 1,
        'phase_step': 10,  # 每个相位的持续时间
        'model': 'DQN',
        'batch_size': 32
    }

    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    intersection_id = list(config['lane_phase_info'].keys())  # all intersections
    config["intersection_id"] = intersection_id
    phase_list = {id_: config["lane_phase_info"][id_]["phase"] for id_ in intersection_id}
    config["phase_list"] = phase_list

    model_dir = "model/{}_{}".format(config['model'], date)
    result_dir = "result/{}_{}".format(config['model'], date)
    config["result_dir"] = result_dir

    # make dirs
    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists("result"):
        os.makedirs("result")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    env = CityFlowEnvM(config["lane_phase_info"],
                       intersection_id,
                       num_step=config["num_step"],
                       thread_num=8,
                       cityflow_config_file=config["cityflow_config_file"]
                       )

    config["state_size"] = env.state_size

    episode_reward = {id_: 0 for id_ in intersection_id}  # for every agent
    episode_score = {id_: 0 for id_ in intersection_id}  # for everg agent
    env.reset()
    for i in range(config['num_step']):
        env.eng.next_step()
        score = env.get_score()
        reward = env.get_reward()
        for id_ in config["intersection_id"]:
            episode_reward[id_]+=reward[id_]
            episode_score[id_]+=score[id_]
        if(i%100==0):
            print(i)

    # for id_ in config['intersection_id']:
    #     episode_reward[id_] /= config['num_step']
    #     episode_score[id_] /= config['num_step']

    print_episode_reward = {'_'.join(k.split('_')[1:]): v for k, v in episode_reward.items()}
    print_episode_score = {'_'.join(k.split('_')[1:]): v for k, v in episode_score.items()}
    print('\n')
    print("Reward:{}, Score: {}".format(print_episode_reward, print_episode_score))
    print('travel time:',env.eng.get_average_travel_time())

if __name__ == '__main__':
    main()
