import chess
import numpy as np

from .move_encoding import utils, queenmoves, knightmoves, underpromotions

class BoardHistory:
    """ Maintains history of recent board positions, encoded as numpy arrays
    The history only retains the k most recent board positions; older positions are discarded when
    new ones are added.
    An array view of the history can be obtained via the view() function
    """
    def __init__(self, length):
        # Ring buffer of recent board encodings; 
        # stored boards are always oriented towards the white player
        self._buffer = np.zeros((length, 8, 8, 14), dtype = np.int64)

    def push(self, board):
        # adds new board to history
        board_array = self.encode(board)

        # overwrite oldest element in the buffer
        self._buffer[-1] = board_array

        # roll inserted element to the top (= most recent position)
        # all older elements are pushed towards the end of the buffer
        self._buffer = np.roll(self._buffer, 1, axis = 0)

    def encode(self, board):
        # converts chess.Board to np.array representation
        array = np.zeros((8, 8, 14), dtype = np.int64)

        for square, piece in board.piece_map().items():
            rank, file = chess.square_rank(square), chess.square_file(square)
            piece_type, color = piece.piece_type, piece.color
        
            # The first 6 planes encode the pieces of the active player, 
            # the following 6 those of the active player's opponent. 
            # Since this class always stores boards oriented towards the white player,
            # white is considered to be the active player here.
            offset = 0 if color == chess.WHITE else 6
            
            # Chess enumerates piece types beginning with one, which we have
            # to account for
            idx = piece_type - 1
        
            array[rank, file, idx + offset] = 1

        # Repetition counters
        array[:, :, 12] = board.is_repetition(2)
        array[:, :, 13] = board.is_repetition(3)

        return array

    def view(self, orientation):
        """ Returns array view of the board history
        Returns (8, 8, k * 14) array view of the k most recently added positions

        If less than k positions have been added since the last reset, missing positions are zeroed out.

        By default, positions are oriented towards the white player;
        setting the optional orientation parameter to `chess.BLACK` will reorient the view towards the black player.
        """
        # copy buffer to not let reorientation affect the internal buffer 
        array = self._buffer.copy()

        if orientation == chess.BLACK:
            for board_array in array:

                # Rotate all planes encoding the position by 180 degrees
                rotated = np.rot90(board_array[:, :, :12], k=2)

                # In the buffer, the first six planes encode white's pieces; 
                # swap with the second six planes
                rotated = np.roll(rotated, axis=-1, shift=6)

                np.copyto(board_array[:, :, :12], rotated)

        # Concatenate k stacks of 14 planes to one stack of k * 14 planes
        array = np.concatenate(array, axis=-1)
        return array

    def reset(self):
        # clears history
        self._buffer[:] = 0


class ChessEnv:
    """ Implements board & move encodings from AlphaZero
    ** Board encoding information **
        Converts board states (`chess.Board` instances) from the wrapped `Chess` environment to numpy arrays

        A state is represented as an array of shape (8, 8, 8 * 14 + 7). 
        This can be thought of as stack of 'planes' of size 8 x 8, where each of the 8 * 14 + 7 planes 
            represent different aspect of the game.

        The first 8 * 14 planes encode the 8 most recent board positions, here referred to as the
            board's 'history', grouped into sets of 14 planes per position.

        The first 6 planes in a set of 14 denote the pices of the active player (the player which moves next). 
            Each of the 6 planes is associated w/ a particular piece type (pawn, knight, bishop, queen, king). 
        The next 6 places encode the pices of the opponent player, following the same scheme. 
        The final 2 planes are binary (all 1's or 0's), and indicate 2-fold and 3-fold repetition of the 
            encoded position over the course of the current game.
        Note that in each step, all board representations in the history are reoriented to the perspective 
            of the active player. 

        The remaining 7 planes encode meta-information about the game state:
            the color of the active player 
            the total move count  
            castling rights for the active player and their opponent (kingside and queenside, respectively) 
            halfmove-clock 


    ** Move encoding information **
        Provides an integer action to the wrapped `Chess` environment.

        Moves are encoded as indices into a flattened 8 x 8 x 73 array, where each position encodes a
            possible move.
        The first 2 dimensions correspond to the square from which the pice is picked up. 
        The last dimension denotes the "move type", which describes how the selected piece is moved from
            its current position.

        AlphaZero defined 3 move types:
            - queen moves: move the piece horizontally, vertically, or diagonally, for any number of
                  squares
            - knight moves: move the piece in an L-shape, i.e. two squares either 
                  horizontally or vertically, followed by one square in the orthogonal
                  direction
            - underpromotions: let a pawn move from the 7th to the 8th rank and
                  promote the piece to either a knight, bishop or rook. 
                  Moving a pawn to the 8th rank with a queen move is automatically assumed to be a queen
                  promotion. 
                  Note that there is no way to not promote a pawn arriving at the opponent's home rank. 
        The 3 moves together capture all possible moves that can be made in the game of chess.
            Castling is represented as a queen move of the king to its left / right by 2 squares.
            There are 56 possible queen moves (8 directions * 7 max; distance of 7 squares)
            8 possible knight moves 
            9 possible underpromotions

        Note that moves are encoded from the current player's perspective.
            Moves for the black are encoded as if they were made by the white player on a rotated board.
    """
    def __init__(self):
        self._observation_space = None
        self._action_space = None

        self._board = None
        self._ready = False
            # indicates if the board has been reset() since it has been created 
            # or the previous game has ended
        self._reward_range = (-1, 1)
        self._rewards = {
                '*':        0.0, # Game not over yet
                '1/2-1/2':  0.0, # Draw
                '1-0':     +1.0, # white wins
                '0-1':     -1.0, # black wins
                }

        self._board_history = BoardHistory(8)

    def reset(self):
        self._observation_space = np.zeros(
                shape = (8, 8, 8 * 14 + 7), 
                dtype = np.int64
        )
        self._action_space = np.zeros(
                shape = 8 * 8 * (8 * 7 + 8 + 9),
                dtype = np.int64
        )

        self._board = chess.Board()
        self._read = True

        self._board_history.reset()

        return self._observation(self._board.copy())

    def step(self, action):
        # assert self._ready, "Cannot call env.step() before calling reset()"

        action = self._decode_move(action)
        if action not in self._legal_moves():
            raise ValueError(
                    f'Illegal move {action} for board position {self._board.fen()}'
            )

        self._board.push(action)

        next_state = self._observation(self._board.copy())
        reward = self._reward()
        done = self._board.is_game_over() or self._board.can_claim_fifty_moves() # or self._board.can_claim_threefold_repitition()
        self._read = not done

        return next_state, reward, done, None

    def _observation(self, board):
        # Converts `chess.Board` instances to numpy arrays 
        self._board_history.push(board)
            # adds the board position to history

        history = self._board_history.view(orientation = board.turn)

        meta = np.zeros(
                shape = (8, 8, 7),
                dtype = np.int64
        )

        # Active player color
        meta[:, :, 0] = int(board.turn)
    
        # Total move count
        meta[:, :, 1] = board.fullmove_number

        # Active player castling rights
        meta[:, :, 2] = board.has_kingside_castling_rights(board.turn)
        meta[:, :, 3] = board.has_queenside_castling_rights(board.turn)
    
        # Opponent player castling rights
        meta[:, :, 4] = board.has_kingside_castling_rights(not board.turn)
        meta[:, :, 5] = board.has_queenside_castling_rights(not board.turn)
    
        # No-progress counter
        meta[:, :, 6] = board.halfmove_clock

        state = np.concatenate([history, meta], axis=-1)
        return state 


    def _legal_moves(self):
        # assert self._ready, "Cannot compute legal moves before calling reset()"

        return self._board.legal_moves

    def _legal_actions(self):
        # assert self._ready, "Cannot compute legal actions before calling reset()"

        return list(self._encode_move(move) for move in self._board.legal_moves)

    def _encode_move(self, move):
        # converts `chess.Move` instance to corresponding integer action for current board position
        if self._board.turn == chess.BLACK:
            move = utils.rotate(move)

        # try to encode given move as queen move, knight move, or underpromotion. 
        # if `move` is not of the associated move type, `encode` func in respective helper modules will return None
        action = queenmoves.encode(move)
        if action is None:
            action = knightmoves.encode(move)
        if action is None:
            action = underpromotions.encode(move)

        # if move doesn't belong to any move type, consider it as invalid
        if action is None:
            raise ValueError(f'{move} is not a valid move')

        return action

    def _decode_move(self, action):
        # converts an integer action to the corresponding `chess.Move` object for the current board
        # position
        move = queenmoves.decode(action)
        is_queen_move = move is not None
        if not move:
            move = knightmoves.decode(action)
        if not move:
            move = underpromotions.decode(action)
        if not move:
            raise ValueError(f'{action} is not a valid action')

        # actions encode moves from the perspective of the current player. 
        # if this is the black player, move must be reoriented
        turn = self._board.turn
        if turn == chess.BLACK:
            move = utils.rotate(move)

        # moving a pawn to the opponent's home rank with a queen move is automatically assumed to be
        # queen underpromotion. 
        # However, since queenmoves have no reference to the board and can thus not determine
        # whether the moved piece is a pawn, we have to add this info manually here
        if is_queen_move:
            to_rank = chess.square_rank(move.to_square)
            is_promoting_move = (
                (to_rank == 7 and turn == chess.WHITE) or 
                (to_rank == 0 and turn == chess.BLACK)
            )


            piece = self._board.piece_at(move.from_square)
            is_pawn = piece.piece_type == chess.PAWN

            if is_pawn and is_promoting_move:
                move.promotion = chess.QUEEN

        return move

    def _reward(self):
        # TODO: upgrade reward system
        result = self._board.result()
        reward = self._rewards[result]
        return reward

    def render(self, mode = None):
        if mode == 'svg':
            return chess.svg.board(self._board)

        return self._board




