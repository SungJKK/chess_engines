import random
from collections import deque
    # use deque as experience replay memory

import gym
import gym_chess
import chess

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, Conv3D, Flatten
from tensorflow.keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.get_logger().setLevel('INFO')

class Agent:
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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # exploration (pick random action)
            return env.legal_actions[random.randrange(0, len(env.legal_actions))]

        # exploitation (invoke Q-network to make prediction)
        act_values = self.q_network.predict(state)[0][0]
        legal_move = act_values[env.legal_actions]
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

# Interact with environment
env = gym.make('ChessAlphaZero-v0')

state_size = 14 * 8 + 7
    # TODO: fix to 8 * 8 * 119 = 7,616
action_size = 8 * 8 * (8 * 7 + 8 + 9)
    # = 4,672
    # refer to AlphaZero paper for representation of the board & moves
agent = Agent(state_size, action_size)

batch_size = 30
    # batch_size for gradient descent
num_of_episodes = 5
    # number of games for our agent to play
timesteps_per_episode = 10
    # if game doesn't end, maximum game time to train

for e in range(0, num_of_episodes):
    # reset the environment
    state = env.reset()
    
    print('============================ Start game ========================')
    print(env.render(mode='unicode'))
    print('================================================================')

    for timestep in range(timesteps_per_episode):
        # run action by agent
        action = agent.act(state)
        
        # take action by environment
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        
        print('================================================================')
        print(env.render(mode='unicode'))
        print('================================================================')

        if done:
            print('Episode: {}/{}, score: {}, e: {:.2}'.format(e, num_of_episodes, timestep, agent.epsilon))
                # check exploration vs. exploitation rate over time 
                # if our agent is not performing well, good place to look is the epsilon
            agent.align_target_model()
                # move weights from q_network to target network
            break

    if batch_size < len(agent.experience_replay):
        agent.replay(batch_size)
        
