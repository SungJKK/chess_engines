import os

from engines import model
from chess_environment import environment

import gym 
import gym_chess

# Interact with environment
# env = gym.make('ChessAlphaZero-v0')
env = environment.ChessEnv()

state_size = 14 * 8 + 7
    # TODO: fix to 8 * 8 * (14 * 8 + 7) = 7,616
action_size = 8 * 8 * (8 * 7 + 8 + 9)
    # = 4,672
    # refer to AlphaZero paper for representation of the board & moves
agent = model.DQN_Agent(state_size, action_size)

batch_size = 30
    # batch_size for gradient descent
num_of_episodes = 5
    # number of games for our agent to play
timesteps_per_episode = 10
    # if game doesn't end, maximum game time to train

for e in range(0, num_of_episodes):
    # reset the environment
    state = env.reset()
    
    # print('============================ Start game ========================')
    # print(env.render(mode='unicode'))
    # print('================================================================')

    for timestep in range(timesteps_per_episode):
        # run action by agent
        action = agent.act(state, env)
        
        # take action by environment
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        
        # print('================================================================')
        # print(env.render(mode='unicode'))
        # print('================================================================')

        if done:
            print('Episode: {}/{}, score: {}, e: {:.2}'.format(e, num_of_episodes, timestep, agent.epsilon))
                # check exploration vs. exploitation rate over time 
                # if our agent is not performing well, good place to look is the epsilon
            agent.align_target_model()
                # move weights from q_network to target network
            break

    if batch_size < len(agent.experience_replay):
        agent.replay(batch_size)

    ''' save & load models for training
    path_dir = './saved_models'
    if e == 2:
        agent.save(path_dir)
    elif e == 4:
        agent.load(path_dir)
    '''

