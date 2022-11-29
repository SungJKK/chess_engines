# import ChessEnvironment 
import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQN_Agent:
    def __init__(self, state_size, action_size):
        self._state_size = state_size
            # state size: 8 * 8 * 119
        self._action_size = action_size
            # action size: 8 * 8 * (8 * 7 + 8 + 9),
            # refer to AlphaZero paper
        self._optimizer = Adam(learning_rate = 0.01)

        self.experience_replay = deque(maxlen = 2000)

        self.gamma = 0.95
        self.epsilon = 0.7
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()

    def _build_compile_model(self): 
        # TODO: create separate class for neural network structure
        model = Sequential([
            # input layer
            Dense(5, input_shape = (8, 119), activation = 'relu'),
            # hidden layers
            Dense(50, activation = 'relu'),
            # output layer
            Dense(self._action_size, activation = 'linear')
        ])

        # compile & return model
        model.compile(loss = 'mse', optimizer = self._optimizer)
        
        return model 

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.experience_replay.append((state, action, reward, next_state, done))

    def act(self, state, env):
        legal_actions = env._legal_actions()

        if np.random.rand() <= self.epsilon:
            # exploration (pick random action)
            return legal_actions[random.randrange(0, len(legal_actions))]

        # exploitation (invoke Q-network to make prediction)
        act_values = self.q_network.predict(state)[0][0]
        legal_move = act_values[legal_actions]
        move = np.argmax(legal_move)
        return np.where(act_values == legal_move[move])[0][0]


    def replay(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.q_network.predict(state)
            if done:
                target[0][0][action] = reward
            else:
                t = self.target_network.predict(next_state)[0][0]
                target[0][0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs = 1, verbose = 0)

        if self.epsilon_min < self.epsilon:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        print('Saving models...')
        self.q_network.save(os.path.join(path, 'q_network.h5'))
        self.target_network.save(os.path.join(path, 'target_network.h5'))

    def load(self, path):
        print('Loading models...')
        self.q_network = tf.keras.models.load_model(os.path.join(path, 'q_network.h5'))
        self.target_network = tf.keras.models.load_model(os.path.join(path, 'target_network.h5'))

        
