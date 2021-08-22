import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DQNAgent(object):
    def __init__(self,
                 intersection_id,
                 state_size=9,
                 action_size=8,
                 batch_size=32,
                 phase_list=[],
                 timing_list = [],
                 env=None
                 ):
        self.env = env
        self.intersection_id = intersection_id
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = 2000
        self.memory = deque(maxlen=self.memory_size)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.learning_rate = 0.001
        self.step = 0
        self.batch_size = batch_size
        self.model = self._build_model()

        self.phase_list = phase_list
        self.timing_list = timing_list

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        action = self.phase_list.index(action)  # index
        self.memory.append((state, action, reward, next_state))

    def remember_timing(self, state, timing, reward, next_state):
        timing = self.timing_list.index(timing)  # index
        self.memory.append((state, timing, reward, next_state))

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon - self.step * 0.0002:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self): # Timing if flag for agent_timing
        if len(self.memory) < self.batch_size:
            return
        self.step + 1
        replay_batch = random.sample(self.memory, self.batch_size)
        s_batch = np.reshape(np.array([replay[0] for replay in replay_batch]), [self.batch_size, self.state_size])
        next_s_batch = np.reshape(np.array([replay[3] for replay in replay_batch]),
                                      [self.batch_size, self.state_size])

        Q = self.model.predict(s_batch)
        Q_next = self.model.predict(next_s_batch)

        lr = 1
        for i, replay in enumerate(replay_batch):
            _, a, reward, _ = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + self.gamma * np.amax(Q_next[i]))

        # 传入网络训练
        # print("s_batch:\n", s_batch, "Q:\n", Q)
        self.model.fit(s_batch, Q, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        # print("model saved:{}".format(name))


class MDQNAgent(object):
    def __init__(self,
                 intersection,
                 state_size=17,
                 batch_size=32,
                 phase_list={},
                 env=None):

        self.intersection = intersection
        self.agents = {}
        self.make_agents(intersection, state_size, batch_size, phase_list, env)

    def make_agents(self, intersection, state_size, batch_size, phase_list, env):
        for id_ in self.intersection:
            self.agents[id_] = DQNAgent(id_,
                                        state_size=state_size,
                                        action_size=len(phase_list[id_]),
                                        batch_size=batch_size,
                                        phase_list=phase_list[id_],
                                        env=env
                                        )

    def remember(self, state, action, reward, next_state):
        for id_ in self.intersection:
            self.agents[id_].remember(state[id_],
                                      action[id_],
                                      reward[id_],
                                      next_state[id_])

    def choose_action(self, state):
        action = {}
        for id_ in self.intersection:
            action[id_] = self.agents[id_].choose_action(state[id_])
        return action

    def replay(self):
        for id_ in self.intersection:
            self.agents[id_].replay()

    def avg_model_weights(self):
        weights = []
        for id_ in self.intersection:
            weights.append(self.agents[id_].model.get_weights())
        avg_weight = np.average(weights, axis=0)

        for id_ in self.intersection:
            self.agents[id_].model.set_weights(avg_weight)

    def load(self, name):
        for id_ in self.intersection:
            assert os.path.exists(name + '.' + id_), "Wrong checkpoint, file not exists!"
            self.agents[id_].load(name + '.' + id_)

    def save(self, name):
        for id_ in self.intersection:
            self.agents[id_].save(id_ + '.' + name)
