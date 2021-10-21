import cityflow
import pandas as pd
import os
from time import time
import math
import numpy as np
import itertools



class CityFlowEnvM(object):
    '''
    multi inersection cityflow environment
    '''

    def __init__(self,
                 lane_phase_info,
                 intersection_id,
                 num_step=2000,
                 thread_num=1,
                 cityflow_config_file='example/config_1x2.json',
                 path_to_log='result',
                 dataset = '1_3',
                 ):
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)
        self.num_step = num_step
        self.intersection_id = intersection_id  # list, [intersection_id, ...]
        self.state_size = None
        self.lane_phase_info = lane_phase_info  # "intersection_1_1"

        self.path_to_log = path_to_log
        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.lane_mapping = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {}  # {id_:[lanes]}
        self.dataset = dataset
        self.lane_intensity = {}
        distances = self.eng.get_vehicle_distance()

        initial_phase = {}
        for id_ in self.intersection_id:
            self.start_lane[id_] = self.lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = self.lane_phase_info[id_]['end_lane']

            # print(self.start_lane[id_])
            # print(self.end_lane[id_])
            # print("\n")

            self.lane_mapping[id_] = self.lane_phase_info[id_]["lane_mapping"]
            # print(self.lane_mapping[id_])
            self.phase_startLane_mapping[id_] = self.lane_phase_info[id_]["phase_startLane_mapping"]

            self.phase_list[id_] = self.lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0
            initial_phase[id_] = 1
            self.lane_intensity[id_] = {}
            self.lane_intensity[id_]['start'] = [0 for _ in self.start_lane[id_]]
            self.lane_intensity[id_]['end'] = [0 for _ in self.end_lane[id_]]
        self.get_state(initial_phase)  # set self.state_size

    def reset(self):
        self.eng.reset()

    def step(self, action_phase, cur_step):
        '''
        action: {intersection_id: phase, ...}
        '''
        t1 = time()

        for id_, a in action_phase.items():
            if self.current_phase[id_] == a:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = a
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_])  # set phase of traffic light
        self.eng.next_step()
        if cur_step % 5 == 4:
            distances = self.eng.get_vehicle_distance()
            for id_ in self.intersection_id:
                self.lane_intensity[id_]['start'] = [self.get_lanepressure(id_, lane, distances) for lane in self.start_lane[id_]]
                self.lane_intensity[id_]['end'] = [self.get_lanepressure(id_, lane, distances) for lane in self.end_lane[id_]]
        reward = self.get_reward()

        simu_time = time() - t1 # simulation time of one step (run one step and return the reward)
        return self.get_state(action_phase), reward, simu_time

    def get_state(self, action_phase = None):
        # print(action_phase)
        state = {id_: self.get_state_(id_, action_phase[id_]) for id_ in self.intersection_id}
        return state

    # get lane pressure with each vehicle
    def get_lanepressure(self, id_, lane, distances):
        L = 300
        if self.dataset == 'jinan':
            if lane[-3] == '0' or lane[-3] == '2':
                L = 400
            else:
                L = 800
        if self.dataset == 'hangzhou':
            if lane[-3] == '0' or lane[-3] == '2':
                L = 800
            else:
                L = 600
        lane_pressure = 0
        sigma = 1.5
        max_speed = 11.111
        vehicles = self.eng.get_lane_vehicles()[lane]
        # print("vehicles:",vehicles)
        if lane in self.start_lane[id_]:
            for v in vehicles:
                x = distances[v]
                lane_pressure += float(format(math.log(
                x / L * sigma * (max_speed - float(self.eng.get_vehicle_info(v)["speed"])) / (
                        float(self.eng.get_vehicle_info(v)["speed"]) + 1) + 1), '.4f'))
                # lane_pressure += float(format(math.log(
                # sigma * (max_speed - float(self.eng.get_vehicle_info(v)["speed"])) / (
                #         float(self.eng.get_vehicle_info(v)["speed"]) + 1) + 1), '.4f'))
                # print("speed for", v, ":", self.eng.get_vehicle_info(v)["speed"],"pressure:",format(math.log(w1*float(self.eng.get_vehicle_info(v)["speed"]) + w2*max_speed + 1),'.4f'))
        else:
            for v in vehicles:
                x = distances[v]
                lane_pressure += float(format(math.log(
                (L - x) / L * sigma * (max_speed - float(self.eng.get_vehicle_info(v)["speed"])) / (
                        float(self.eng.get_vehicle_info(v)["speed"]) + 1) + 1), '.4f'))
                # lane_pressure += float(format(math.log(
                # sigma * (max_speed - float(self.eng.get_vehicle_info(v)["speed"])) / (
                #         float(self.eng.get_vehicle_info(v)["speed"]) + 1) + 1), '.4f'))
                # print("speed for", v, ":", self.eng.get_vehicle_info(v)["speed"],"pressure:",format(math.log(w1*float(self.eng.get_vehicle_info(v)["speed"]) + w2*max_speed + 1),'.4f'))
        return lane_pressure

    def get_neigh_pressure(self, nei_id_, row, col, nei_row, nei_col, action_phase = None):
        pressure = 0
        distances = self.eng.get_vehicle_distance()
        start_vehicle_count = self.lane_intensity[nei_id_]['start']

        if nei_row < row: # left, pressure comes from turn-right(start_vehicle_count[5]) AND (WE(start_vehicle_count[1]) OR NSL(start_vehicle_count[6]))
            if action_phase == 1: # WE
                pressure += start_vehicle_count[5] + start_vehicle_count[1]
            if action_phase == 4:  # NSL
                pressure += start_vehicle_count[5] + start_vehicle_count[6]

        if nei_row > row: # right, pressure comes from turn-right(start_vehicle_count[8]) AND (WE(start_vehicle_count[10]) OR NSL(start_vehicle_count[3]))
            if action_phase == 1: # WE
                pressure += start_vehicle_count[8] + start_vehicle_count[10]
            if action_phase == 4:  # NSL
                pressure += start_vehicle_count[8] + start_vehicle_count[3]

        if nei_col < col: # above, pressure comes from turn-right(start_vehicle_count[11]) AND (NS(start_vehicle_count[4]) OR WEL(start_vehicle_count[0]))
            if action_phase == 2: # NS
                pressure += start_vehicle_count[11] + start_vehicle_count[4]
            if action_phase == 3:  # WEL
                pressure += start_vehicle_count[11] + start_vehicle_count[0]

        if nei_col > col: # below, pressure comes from turn-right(start_vehicle_count[2]) AND (NS(start_vehicle_count[7]) OR WEL(start_vehicle_count[9]))
            if action_phase == 2: # NS
                pressure += start_vehicle_count[2] + start_vehicle_count[7]
            if action_phase == 3:  # WEL
                pressure += start_vehicle_count[2] + start_vehicle_count[9]

        return pressure

    def get_state_(self, id_, action_phase = None):
        inters = str.split(id_, '_')
        row = int(inters[1])
        column = int(inters[2])
        neighbor = [] # neighbor intersection
        neighbor.append([row - 1, column])
        neighbor.append([row, column - 1])
        neighbor.append([row + 1, column])
        neighbor.append([row, column + 1])
        eta = 0.1 # discount ctor of neighbour's pressure
        state = self.intersection_info(id_)
        pressure = []
        temp = []
        distances = self.eng.get_vehicle_distance()

        # 对应车道求压力差
        end_vehicle_count_avg = []
        for i in range(4):
            end_vehicle_count_avg.append(math.ceil(sum([self.lane_intensity[id_]['end'][j] for j in range(i * 3, i * 3 + 3)]) / 3))
        # print(start_vehicle_count)
        # print(end_vehicle_count)
        start_vehicle_count_cop = []
        index = [1, 0, 2, 1, 0, 3, 3, 2]
        for i in range(len(self.lane_intensity[id_]['start'])):
            if i % 3 != 2:
                start_vehicle_count_cop.append(self.lane_intensity[id_]['start'][i])
        # print(start_vehicle_count_cop)
        # print(end_vehicle_count_avg)
        for i in range(len(start_vehicle_count_cop)):
            temp.append(start_vehicle_count_cop[i] - end_vehicle_count_avg[index[i]])

        pressure.append(temp[1] + temp[7])  # straight
        pressure.append(temp[3] + temp[5]) # straight
        pressure.append(temp[0] + temp[6]) # left
        pressure.append(temp[2] + temp[4]) # left

        return_state = pressure + [state['current_phase']] # add id_'s current_phase

        for e in neighbor: # add neghbors' current_phase
            interid = inters[0] + '_' + str(e[0]) + '_' + str(e[1])
            if interid in self.intersection_id:
                return_state.append(eta * self.get_neigh_pressure(nei_id_=interid, row=row, col=column, nei_col=e[0], nei_row=e[1], action_phase=action_phase))
            else:
                return_state.append(0)

        return self.preprocess_state(return_state)

    def get_timing_(self, id_, phase):
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]
        start_vehicle_count_cop = []  # 1->1,7; 2->3,5; 3->0,6; 4->2,4
        for i in range(len(start_vehicle_count)):
            if i % 3 != 2:
                start_vehicle_count_cop.append(start_vehicle_count[i])

        phase_lane = [[1,7], [3,5], [0,6], [2,4]]
        w1 = 1
        w2 = 2
        max_count = max(start_vehicle_count_cop[phase_lane[phase-1][0]], start_vehicle_count_cop[phase_lane[phase-1][1]])
        min_count = min(start_vehicle_count_cop[phase_lane[phase-1][0]], start_vehicle_count_cop[phase_lane[phase-1][1]])
        vehicle_count = (min_count * w1 + max_count * w2)/(w1+w2)
        timing = math.ceil(vehicle_count/2)*5
        if timing>25:
            timing = 25
        if timing < 5:
            timing = 5
        return vehicle_count, timing

    def intersection_info(self, id_):
        '''
        info of intersection 'id_'
        '''
        state = {}
        state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()
        state['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()
        state['lane_vehicles'] = self.eng.get_lane_vehicles()
        state['vehicle_speed'] = self.eng.get_vehicle_speed()
        state['start_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.end_lane[id_]}
        state['start_lane_waiting_vehicle_count'] = {lane: state['lane_waiting_vehicle_count'][lane] for lane in
                                                     self.start_lane[id_]}
        state['end_lane_waiting_vehicle_count'] = {lane: state['lane_waiting_vehicle_count'][lane] for lane in
                                                   self.end_lane[id_]}
        state['start_lane_vehicles'] = {lane: state['lane_vehicles'][lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicles'] = {lane: state['lane_vehicles'][lane] for lane in self.end_lane[id_]}
        state['start_lane_speed'] = {
            lane: np.sum(list(map(lambda vehicle: state['vehicle_speed'][vehicle], state['lane_vehicles'][lane]))) / (
                    state['lane_vehicle_count'][lane] + 1e-5) for lane in
            self.start_lane[id_]}  # compute start lane mean speed
        state['end_lane_speed'] = {
            lane: np.sum(list(map(lambda vehicle: state['vehicle_speed'][vehicle], state['lane_vehicles'][lane]))) / (
                    state['lane_vehicle_count'][lane] + 1e-5) for lane in
            self.end_lane[id_]}  # compute end lane mean speed
        state['current_phase'] = self.current_phase[id_]
        state['current_phase_time'] = self.current_phase_time[id_]
        return state

    def preprocess_state(self, state, action = None):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        if action != None:
            return_state = np.reshape(return_state, [1, self.state_size+1])
        else:
            return_state = np.reshape(return_state, [1, self.state_size])

        return return_state

    def get_reward(self):
        reward = {id_: self.get_reward_(id_) for id_ in self.intersection_id}
        return reward

    def get_reward_(self, id_):
        '''
        every agent/intersection's reward
        '''
        distances = self.eng.get_vehicle_distance()
        start_vehicle_count = [self.get_lanepressure(id_, lane, distances) for lane in self.start_lane[id_]]
        end_vehicle_count = self.lane_intensity[id_]['end']
        start_vehicle_count_cop = []
        end_vehicle_count_cop = []
        for i in range(len(start_vehicle_count)):
            if i % 3 != 2:
                start_vehicle_count_cop.append(start_vehicle_count[i])
                end_vehicle_count_cop.append(end_vehicle_count[i])
        pressure = sum(start_vehicle_count_cop) - sum(end_vehicle_count_cop)

        reward = -pressure
        return reward

    def get_pressure(self):
        pressure = {id_: self.get_pressure_(id_) for id_ in self.intersection_id}
        return pressure

    def get_pressure_(self, id_):
        '''
        every agent/intersection's reward
        '''
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]
        pressure = sum(start_vehicle_count) - sum(end_vehicle_count)
        return pressure

    def get_score(self):
        score = {id_: self.get_score_(id_) for id_ in self.intersection_id}
        return score

    def get_score_(self, id_):
        state = self.intersection_info(id_)
        start_lane_waiting_vehicle_count = state['start_lane_waiting_vehicle_count']
        end_lane_waiting_vehicle_count = state['end_lane_waiting_vehicle_count']
        score = -1 * np.sum(
            list(start_lane_waiting_vehicle_count.values()) + list(end_lane_waiting_vehicle_count.values()))
        return score

    def bulk_log(self):
        # self.eng.print_log(os.path.join(self.path_to_log, "replay.txt"))
        self.eng.set_replay_file((os.path.join(self.path_to_log, "replay.txt")))