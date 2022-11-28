import chess

import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D, Dense, MaxPooling3D
from keras.layers import add, BatchNormalization, Flatten
from keras.losses import CategoricalCrossentropy, MeanSquaredError

from board_conversion import *
from variable_settings import *
from rewards import *

# Deep Q-Learning Model
class DQN_Agent():
    def __init__(self, model=None):
        if model:
            print('CUSTOM MODEL SET')
            self.model = model
        else:
            self.model = self.create_q_model()

    def create_q_model(self):
        # network defined by the deepmind paper
        input_layer = Input(shape = (8, 8, 12))
        
        # convultions on the frames on the screen
        # convolutional network w/ 3 convolutional layers w/ relu activation function
        x = Conv2D(filters=64,kernel_size = 2,strides = (2,2),activation = 'relu')(input_layer)
        x = Conv2D(filters=128,kernel_size=2,strides = (2,2),activation = 'relu')(x)
        x = Conv2D(filters=256,kernel_size=2,strides = (2,2),activation = 'relu')(x)
        x = Flatten()(x)
        
        # final layer contains 4096 neurons, representing 4096 possible moves that can be played
        # in any given position (64 for starting square, 64 for ending square)
        action = Dense(4096, activation = 'softmax')(x)
        
        return Model(inputs = input_layer, outputs = action)
    
    def predict(self, env):
        # converts board into a tensor
        # then calls model to predict an output for input
        # outputs a probability distribution that sums up to 1
        # assume each of the values map to a move, and a larger probabilities represent the confidence of hte model
        # then applies a mask that removes all illegal moves from that distribution
        # then move is converted to a chess move via a pre-defined dictionary
        state_tensor = tf.convert_to_tensor(env.translate_board())
        state_tensor = tf.expand_dims(state_tensor, 0)
        
        action_probs = self.model(state_tensor, training = False)
        action_space = filter_legal_moves(env.board, action_probs[0])
        action = np.argmax(action_space, axis = None)
        move = num2move[action]
        return move, action
    
    def explore(self, env):
        # chooses random legal move to play
        # this allows possibility to break out of local minimas, but will slow training progress at beginning
        action_space = np.random.randn(4096)
        action_space = filter_legal_moves(env.board, action_space)
        action = np.argmax(action_space, axis = None)
        move = num2move[action]
        return move, action


# Environment
model = Q_model()
target_model = Q_model()

class ChessEnv():
    ''' The environment of which the model interacts with
    '''
    def __init__(self):
        self.board = chess.Board()
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = {
            'white' : [],
            'black' : [],
        }
        self.done_history = []
        self.episode_reward_history = []
        self.move_counter = 1
        self.fast_counter = 0
        self.pgn = ''
        self.pgns = []
        pass
    
    # Return State
    # returns vector that expresses the information accessible to the agent environment,
    # that can be plugged into the neural network
    def step(self, action):
        # Converts python-chess board object into a matrix
        # converts board into rows & cols and finds each square on the board
        # then it one-hot encodes all the variables in 8x8x12 array
        if self.board.turn:
            turn = 'white'
        else:
            turn = 'black'

        state = self.translate_board()
        rewards = evaluate_reward(self.board,action)
        self.rewards_history['white'].append(rewards[0])
        self.rewards_history['black'].append(rewards[0])
        self.update_pgn(action)
        self.board.push(action)

        state_next = self.board
        state_next = translate_board(state_next)

        self.done = self.board.is_game_over()

        self.action_history.append(move2num[action])
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(self.done)
        self.episode_reward_history.append(rewards)
    
    # Accept Action
    # accepts an action and changes state of environment by this action
    
    # Return Rewards
    # should be a system in place that allows for agent to directly access the rewards from each action
    def translate_board(self):
        pgn = self.board.epd()
        foo = []  
        pieces = pgn.split(" ", 1)[0]
        rows = pieces.split("/")
        for row in rows:
            foo2 = []  
            for thing in row:
                if thing.isdigit():
                    for i in range(0, int(thing)):
                        foo2.append(chess_dict['.'])
                else:
                    foo2.append(chess_dict[thing])
            foo.append(foo2)
        return np.array(foo)


# Execute episode
env = ChessEnv()
for _ in range(iterations):
    state = np.array(env.reset())
    episode_reward = 0
    len_episodes += 1
    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            move,action = model.explore(env)
        else:
            move,action = model.predict(env)
            
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)
        
        env.step(move)
        
        state_samples,masks,updated_q_values = env.update_q_values()
        for i in range(2):
            with tf.GradientTape() as tape:
                q_values = model.model(state_samples.reshape(len(state_samples),8,8,12))
                q_action = tf.reduce_sum(tf.multiply(q_values, masks[i]), axis=1)
                loss = loss_function(updated_q_values[i], q_action)

            grads = tape.gradient(loss, model.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.model.trainable_variables))

        if frame_count % update_target_network == 0:
            model_target.model.set_weights(model.model.get_weights())
            template = "episode {}, frame count {}"
            print(template.format(episode_count, frame_count))
            
        env.episode_reward_history.append(episode_reward)
        if env.done:
            break

    episode_count += 1


# Chess board conversion 
chess_dict = {
    'p': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'n': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'b': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'r': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'q': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'k': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

pos_promo = ['q', 'r', 'b', 'n']
columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
sides = [7, 2]
increments = [1, -1]

promo_moves = []
ucis = []
for side in sides:
    increment = increments[sides.index(side)]
    for i in range(len(columns)):
        current_col = columns[i]
        pos_end_squares = [current_col]
        if i-1 >= 0:
            pos_end_squares.append(columns[i-1])
        if i+1 < len(columns):
            pos_end_squares.append(columns[i+1])

        pos_end_squares = [pos+str(side)for pos in pos_end_squares]
        for promo in pos_promo:
            for end_square in pos_end_squares:
                uci = end_square+current_col+str(side+increment)+promo
                ucis.append(uci)
                move = chess.Move.from_uci(uci)
                promo_moves.append(move)

# ! Replace num2move as list and remove move2num
num2move = []

counter = 0
for from_sq in range(64):
    for to_sq in range(64):
        num2move.append(chess.Move(from_sq, to_sq))
        counter += 1
for move in promo_moves:
    num2move.append(move)
    counter += 1


def generate_side_matrix(board, side):
    matrix = board_matrix(board)
    translate = translate_board(board)
    bools = np.array([piece.isupper() == side for piece in matrix])
    bools = bools.reshape(8, 8, 1)

    side_matrix = translate*bools
    return np.array(side_matrix)


def generate_input(positions, len_positions=8):
    board_rep = []
    for position in positions:
        black = generate_side_matrix(position, False)
        white = generate_side_matrix(position, True)
        board_rep.append(black)
        board_rep.append(white)
    turn = np.zeros((8, 8, 12))
    turn.fill(int(position.turn))
    board_rep.append(turn)

    while len(board_rep) < len_positions*2 + 1:
        value = np.zeros((8, 8, 12))
        board_rep.insert(0, value)
    board_rep = np.array(board_rep)
    board_rep = board_rep[-17:]
    return board_rep


def translate_board(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append(chess_dict['.'])
            else:
                foo2.append(chess_dict[thing])
        foo.append(foo2)
    return np.array(foo)


def board_matrix(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo.append('.')
            else:
                foo.append(thing)
    return np.array(foo)


def translate_move(move):
    from_square = move.from_square
    to_square = move.to_square
    return np.array([from_square, to_square])


def filter_legal_moves(board, logits):
    filter_mask = np.zeros(logits.shape)
    legal_moves = board.legal_moves
    for legal_move in legal_moves:
        from_square = legal_move.from_square
        to_square = legal_move.to_square
        idx = num2move.index(chess.Move(from_square, to_square))
        filter_mask[idx] = 1
    new_logits = logits*filter_mask
    return new_logits
