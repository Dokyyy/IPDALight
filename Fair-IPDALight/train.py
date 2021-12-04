import math
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from time import *
from copy import deepcopy

from cityflow_env import CityFlowEnvM
from utility import *
# from network import MultiLightAgent
from dqn_agent import DQNAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset = "3_3"

    cityflow_config = {
        "interval": 1,
        "seed": 0,
        "laneChange": False,
        "dir": "data/",
        "roadnetFile": "template_lsr/new/" + dataset + "/" + "roadnet_" + dataset + ".json",
        "flowFile": "template_lsr/new/" + dataset + "/" + "syn_" + dataset + "_gaussian_500_1h.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "replayRoadNet.json",
        "replayLogFile": "replayLogFile.txt"
    }


    with open(os.path.join("data/", "cityflow.config"), "w") as json_file:
        json.dump(cityflow_config, json_file)

    config = {
        'cityflow_config_file': "data/cityflow.config",
        'epoch': 200,
        'num_step': 3600,  # 每个epoch的执行步数
        'save_freq': 1,
        'phase_step': 5,  # 每个相位的基础持续时间
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

    timing_list = {id_: [i*5 + config['phase_step'] for i in range(1,5)] for id_ in intersection_id}
    config['timing_list'] = timing_list

    model_dir = "model/{}_{}".format(config['model'], date)
    result_dir = "result/{}_{}".format(config['model'], date)
    config["result_dir"] = result_dir

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
                       cityflow_config_file=config["cityflow_config_file"],
                       dataset=dataset
                       )

    config["state_size"] = env.state_size

    Magents = {}
    for id_ in intersection_id:
        agent = DQNAgent(id_,
                         state_size=config["state_size"],
                         action_size=len(phase_list[id_]),
                         batch_size=config["batch_size"],
                         phase_list=phase_list[id_],
                         timing_list=timing_list[id_],
                         env=env)
        Magents[id_] = agent

    EPISODES = config['epoch']
    total_step = 0
    episode_travel_time = []
    with open(result_dir + "/" + "Travel Time-" + dataset + "-FAIR-IPDALight.csv", 'a+') as ttf:
        ttf.write("travel time\n")
    ttf.close()

    with open(result_dir + "/" + "Max Waiting Time-" + dataset + "-FAIR-IPDALight.csv", 'a+') as ttf:
        ttf.write("max waiting time\n")
    ttf.close()

    with tqdm(total=EPISODES * config['num_step']) as pbar:
        for i in range(EPISODES):
            env.reset()
            state = {}
            action = {}
            action_phase = {}
            timing_phase = {}
            reward = {id_: 0 for id_ in intersection_id}
            rest_timing = {id_: 0 for id_ in intersection_id}
            timing_choose = {id_: [] for id_ in intersection_id}
            pressure = {id_: [] for id_ in intersection_id}
            green_wave = {id_: [] for id_ in intersection_id}
            MAX_WAITING_TIME = 0
            for id_ in intersection_id:
                state[id_] = env.get_state_(id_)

            for episode_length in range(config['num_step']):
                for id_, t in rest_timing.items():
                    if t == 0:
                        if episode_length != 0:
                            # remember and replay the last transition
                            reward[id_] = env.get_reward_(id_)
                            Magents[id_].remember(state[id_], action_phase[id_], reward[id_], next_state[id_])
                            Magents[id_].replay()
                            state[id_] = next_state[id_]

                            # PRESSURE[id_].append(env.get_pressure_(id_))

                        action[id_] = Magents[id_].choose_action(state[id_])
                        action_phase[id_] = phase_list[id_][action[id_]]

                        p, timing_phase[id_] = env.get_timing_(id_, action_phase[id_])
                        rest_timing[id_] = timing_phase[id_]
                        timing_choose[id_].append(timing_phase[id_])
                        pressure[id_].append(p)
                        green_wave[id_].append([action_phase[id_], timing_phase[id_]])


                next_state, reward_ = env.step(action_phase, episode_length)  # one step
                max_time = env.update_waiting_time()
                if max_time > MAX_WAITING_TIME:
                    MAX_WAITING_TIME = max_time

                total_step += 1
                pbar.update(1)
                print_reward = deepcopy(reward)
                pbar.set_description(
                    "t_st:{}, epi:{}, st:{}, mwt:{} r:{}".format(total_step, i + 1, episode_length, MAX_WAITING_TIME,
                                                                 print_reward))

                for id_ in rest_timing:
                    rest_timing[id_] -= 1
                # print("rest_timing: ", rest_timing, "\n")

            episode_travel_time.append(env.eng.get_average_travel_time())
            with open(result_dir + "/" + "Travel Time-" + dataset + "-FAIR-IPDALight.csv", 'a+') as ttf:
                ttf.write("{}\n".format(env.eng.get_average_travel_time()))
            ttf.close()

            with open(result_dir + "/" + "Max Waiting Time-" + dataset + "-FAIR-IPDALight.csv", 'a+') as ttf:
                ttf.write("{}\n".format(MAX_WAITING_TIME))
            ttf.close()

            print('\n')
            print('Epoch {} travel time:'.format(i+1), env.eng.get_average_travel_time())

        df = pd.DataFrame({"travel time": episode_travel_time})
        df.to_csv(result_dir + '/IPDALight.csv', index=False)

        # save figure
        plot_data_lists([episode_travel_time], ['travel time'], figure_name=result_dir + '/travel time.pdf')

        # env.bulk_log()


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    run_time = end_time - start_time
    print('Run time:', run_time)

