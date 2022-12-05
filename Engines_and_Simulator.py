import chess
import random

# Values of all the pieces and pawns, positive for white and negative for black
pieceValue = {"K":0, "Q":9, "R":5, "B":3, "N":3, "P":1,
              "k":0, "q":-9, "r":-5, "b":-3, "n":-3, "p":-1, 
              "None":0}
# Subjective value for checkmate, which should be much greater than any "scoring" of the board.
checkmateVal = 1000

# Piece values based on tables in https://www.chessprogramming.org/Simplified_Evaluation_Function 
# Slighly edited to better match my functions, 8x8 grids written as a 64 length list to represent 
# the additional value a certain piece gets on a certain square

# the library starts the bottom left of the board index 0, but the arrays start top left index 0, so
# the tables may look a little weird, basically first row in the table is the last row on the board
P_table =       [0,  0,  0,  0,  0,  0,  0,  0,
                .05, .10, .10, -.20, -.20, .10, .10,  .05,
                .05, -.05,-.10,  .10,  .10,-.10, -.05,  .05,
                0,  0,  .10, .25, .25,  0,  0,  0,
                .05,  .05, .10, .25, .25, .10,  .05,  .05,
                .10, .10, .20, .30, .30, .20, .10, .10,
                .50, .50, .50, .50, .50, .50, .50, .50,
                0,  0,  0,  0,  0,  0,  0,  0]

p_table =       [0,  0,  0,  0,  0,  0,  0,  0,
                .50, .50, .50, .50, .50, .50, .50, .50,
                .10, .10, .20, .30, .30, .20, .10, .10,
                .05,  .05, .10, .25, .25, .10,  .05,  .05,
                0,  0,  .10, .25, .25,  0,  0,  0,
                .05, -.05,-.10,  0,  0,-.10, -.05,  .05,
                .05, .10, .10, -.20, -.20, .10, .10,  .05,
                0,  0,  0,  0,  0,  0,  0,  0]

p_table = [-x for x in p_table]

B_table =       [-.20, -.10, -.10, -.10, -.10, -.10 ,-.10, -.20,
                -.10,  .05,  0,  0,  0,  0,  .05, -.10,
                -.10, .10, .10, .10, .10, .10, .10, -.10,
                -.10,  0, .10, .10, .10, .10,  0, -.10,
                -.10,  .05,  .05, .10, .10,  .05,  .05, -.10,
                -.10,  0,  .05, .10, .10,  .05,  0, -.10,
                -.10, 0,  0,  0,  0,  0,  0, -.10,
                -.20, -.10, -.10, -.10, -.10, -.10, -.10, -.20]

b_table =       [-.20, -.10, -.10, -.10, -.10, -.10, -.10, -.20,
                -.10, 0,  0,  0,  0,  0,  0, -.10,
                -.10,  0,  .05, .10, .10,  .05,  0, -.10,
                -.10,  .05,  .05, .10, .10,  .05,  .05, -.10,
                -.10,  0, .10, .10, .10, .10,  0, -.10,
                -.10, .10, .10, .10, .10, .10, .10, -.10,
                -.10,  .05,  0,  0,  0,  0,  .05, -.10,
                -.20, -.10, -.10, -.10, -.10, -.10 ,-.10, -.20]

b_table = [-x for x in b_table]

N_table =       [-.50, -.20, -.10, -.10, -.10, -.10, -.20, -.50,
                -.40, -.20,  0,  0,  0,  0, -.20, -.40,
                -.30,  .05, .08, .05, .05, .08,  .05, -.30,
                -.30,  0, .08, .10, .10, .08,  0, -.30,
                -.30,  .05, .10, .10, .10, .10,  .05, -.30,
                -.30,  0, .10, .15, .15, .10,  0, -.30,
                -.40, -.20,  0,  0,  0,  0, -.20, -.40,
                -.50, -.40, -.30, -.30, -.30, -.30, -.40, -.50]

n_table =       [-.50, -.40, -.30, -.30, -.30, -.30, -.40, -.50,
                -.40, -.20,  0,  0,  0,  0, -.20, -.40,
                -.30,  0, .10, .15, .15, .10,  0, -.30,
                -.30,  .05, .10, .10, .10, .10,  .05, -.30,
                -.30,  0, .08, .10, .10, .08,  0, -.30,
                -.30,  .05, .08, .05, .05, .08,  .05, -.30,
                -.40, -.20,  0,  .05,  .05,  0, -.20, -.40,
                -.50, -.20, -.10, -.10, -.10, -.10, -.20, -.50]

n_table = [-x for x in n_table]

R_table =      [0,  0,  0,  .05,  .05,  0,  0,  0,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               .05, .10, .10, .10, .10, .10, .10, .5,
               0,  0,  0,  0,  0,  0,  0,  0]

r_table =      [0,  0,  0,  0,  0,  0,  0,  0,
               .05, .10, .10, .10, .10, .10, .10, .5,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               0,  0,  0,  .05,  .05,  0,  0,  0]

r_table = [-x for x in r_table]

Q_table =       [-.20,-.10,-.10, -.05, -.05,-.10,-.10,-.20,
                -.10,  0,  .05,  0,  0,  0,  0,-.10,
                -.10,  .05,  .05,  .05,  .05,  .05,  0,-.10,
                  0,  0,  .05,  .05,  .05,  .05,  0, -.05,
                  -.05,  0,  .05,  .05,  .05,  .05,  0, -.05,
                -.10,  0,  .05,  .05,  .05,  0,  0,-.10,
                -.10,  0,  0,  0,  0,  0,  0,-.10,
                -.20,-.10,-.10, -.05, -.05,-.10,-.10,-.20]

q_table =       [-.20,-.10,-.10, -.05, -.05,-.10,-.10,-.20,
                -.10,  0,  0,  0,  0,  0,  0,-.10,
                -.10,  0,  .05,  .05,  .05,  0,  0,-.10,
                 -.05,  0,  .05,  .05,  .05,  .05,  0, -.05,
                  0,  0,  .05,  .05,  .05,  .05,  0, -.05,
                -.10,  .05,  .05,  .05,  .05,  .05,  0,-.10,
                -.10,  0,  .05,  0,  0,  0,  0,-.10,
                -.20,-.10,-.10, -.05, -.05,-.10,-.10,-.20]

q_table = [-x for x in q_table]


K_table =       [.20, .30, .10,  0,  0, .10, .30, .20,
                .20, .20,  0,  0,  0,  0, .20, .20,
                -.10,-.20,-.20,-.20,-.20,-.20,-.20,-.10,
                -.20,-.30,-.30,-.40,-.40,-.30,-.30,-.20,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30]

k_table =       [-.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.20,-.30,-.30,-.40,-.40,-.30,-.30,-.20,
                -.10,-.20,-.20,-.20,-.20,-.20,-.20,-.10,
                .20, .20,  0,  0,  0,  0, .20, .20,
                .20, .30, .10,  0,  0, .10, .30, .20]

k_table = [-x for x in k_table]
 
pieceTable = {"P": P_table, "p": p_table, "B": B_table, "b": b_table, "N": N_table, "n": n_table,
              "R": R_table, "r": r_table, "Q": Q_table, "q": q_table, "K": K_table, "k": k_table,
              "None": [0] * 64}

# Basic scoring algorithm, simply sums up all the piece values for the pieces on the board
def scoreBoard(board):
    score = 0
    for square in chess.SQUARES:
        score = score + pieceValue[str(board.piece_at(square))]
    return score

# Same as scoreBoard function, but includes the bonus value of pieces using the 8x8 score tables
def improvedScoreBoard(board):
    score = 0
    for square in chess.SQUARES:
        piece = str(board.piece_at(square))
        score = score + pieceValue[piece] + pieceTable[piece][square]
    return score

# Outputs a random move, used for testing
def randomMove(board):
    legal_moves = list(board.legal_moves)
    return legal_moves[random. randint(0,len(legal_moves) - 1)]

# Outputs the first move in the list, used for testing
def firstMove(board):
    return list(board.legal_moves)[0]

# Checkmate, check, capture, random move (CCCR) algorithm, prioritizes moves in that order
def CCCR(board):
    CCCRmove = randomMove(board)
    score = -1
    for move in board.legal_moves:
        currentScore = 0
        if(board.is_capture(move)):
            currentScore = 1
        board.push(move)
        if(board.is_check()):
            currentScore = 2
        if(board.is_checkmate()):
            currentScore = 3
        if(currentScore > score):
            score = currentScore
            CCCRmove = move
        board.pop()
    return CCCRmove

# Finds the move that causes the highest score after playing it (like taking a piece)
def greedyMove(board):
    bestMove = randomMove(board)
    turnVal = 1
    currentScore = 0
    if(not board.turn):
        turnVal = -1;
    maxEval = -checkmateVal
    for move in board.legal_moves:
        board.push(move)
        if(board.is_checkmate()):
            currentScore = checkmateVal
        else: 
            currentScore = turnVal * scoreBoard(board)
        if(currentScore > maxEval):
            maxEval = currentScore
            bestMove = move
        board.pop()
    return bestMove

# Maximizes score by minimizing opponent score (considers opponents move)
# Should work for both sides
# Randomly shuffle since the first move will always be the same if not shuffled, due to
# the way scoring works with the function (won't change for ties, and same first values)
def twomove_minimax(board):
    bestMove = randomMove(board)
    turn = 1
    if(not board.turn):
        turn = -1;
    maxEval = checkmateVal
    legalmoves = list(board.legal_moves)
    random.shuffle(legalmoves)
    for move in legalmoves:
        opp_maxEval = -checkmateVal
        board.push(move)
        if(board.is_checkmate()):
            board.pop();
            return move;
        nextmoves = list(board.legal_moves)
        random.shuffle(nextmoves)
        for nextmove in nextmoves:
            Eval = 0
            board.push(nextmove)
            if(board.is_checkmate()):
                Eval = checkmateVal
            else: 
                Eval = -turn * scoreBoard(board)
            if(Eval > opp_maxEval):
                opp_maxEval = Eval
            board.pop()
        if(maxEval > opp_maxEval):
            maxEval = opp_maxEval 
            bestMove = move
        board.pop()
    return bestMove

# same as twomove_minimax but at a depth (can set to any number of moves)
def basic_minimax(board, depth):
    if depth == 0 or board.is_game_over():
        return [scoreBoard(board), None]
    if board.turn:
        maxvalue = -checkmateVal
        legalmoves = list(board.legal_moves)
        random.shuffle(legalmoves)
        for move in legalmoves:
            board.push(move)
            if(board.is_checkmate()):
                value = checkmateVal
            else:
                value = basic_minimax(board, depth-1)[0]
            if(value > maxvalue):
                maxvalue = value
                bestmove = move
            board.pop()
        return [maxvalue, bestmove]
    else:
        minvalue = checkmateVal
        legalmoves2 = list(board.legal_moves)
        random.shuffle(legalmoves2)
        for move in legalmoves2:
            board.push(move)
            if(board.is_checkmate()):
                value = -checkmateVal
            else:
                value = basic_minimax(board, depth-1)[0]
            if(value < minvalue):
                minvalue = value
                bestmove = move
            board.pop()
        return [minvalue, bestmove]
    
    
# A lot of improvements compared to basic_minimax: 
# Looks for checkmate, uses an improved scoring system (explained near the function), 
# optimize timing through alpha beta pruning (able to reach depth 4, slow at 5)
# alpha-beta pruning https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
# With this function, we need to return an array, since need to keep track of the 
# current best move as well as the current max/min, alpha, and beta value
# Works for white 
def improved_minimax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return [improvedScoreBoard(board), None]
    if board.turn:
        maxvalue = -checkmateVal
        legalmoves = list(board.legal_moves)
        random.shuffle(legalmoves)
        for move in legalmoves:
            board.push(move)
            if(board.is_checkmate()):
                value = checkmateVal
            else:
                value = improved_minimax(board, depth-1, -beta, -alpha)[0]
            if(value > maxvalue):
                maxvalue = value
                bestmove = move
            board.pop()
            if(maxvalue > alpha):
                alpha = maxvalue
            if(alpha >= beta):
                break
        return [maxvalue, bestmove]
    else:
        minvalue = checkmateVal
        legalmoves2 = list(board.legal_moves)
        random.shuffle(legalmoves2)
        for move in legalmoves2:
            board.push(move)
            if(board.is_checkmate()):
                value = -checkmateVal
            else:
                value = improved_minimax(board, depth-1, -beta, -alpha)[0]
            if(value < minvalue):
                minvalue = value
                bestmove = move
            board.pop()
            if(minvalue < beta):
                beta = minvalue
            if(alpha >= beta):
                break
        return [minvalue, bestmove]

# Previous function only worked for white, this one works for both sides, and is
# much simplier by using turn values similar to twomove_minimax, and by focusing solely
# on maximizing the magnitude rather than maximizing/minimizing positive/negative values
def final_minimax(board, depth, alpha, beta, maximizingWhite):
    turn = 1
    bestMove = randomMove(board)
    if not maximizingWhite:
        turn = -1;
    if depth == 0 or board.is_game_over():
        return [turn * improvedScoreBoard(board), None]
    maxEval = -checkmateVal
    for move in board.legal_moves:
        board.push(move)
        if(board.is_checkmate()):
            Eval = turn * checkmateVal
        elif(board.is_stalemate() or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
            Eval = 0
        else:
            Eval = -final_minimax(board, depth-1, -beta, -alpha, not maximizingWhite)[0]
        board.pop()
        if(Eval > maxEval):
            maxEval = Eval
            bestMove = move
        alpha = max(alpha, maxEval)
        if(alpha >= beta):
            break
    return [maxEval, bestMove]
        

# Below is code to run a game simulation
gameOver = False
board=chess.Board()
print(improvedScoreBoard(board))
while not gameOver:
    if board.turn: 
        # Alpha starts off as lowest value (-checkmateVal) and beta as highest 
        # value (checkmateVal), True since maximizing white
        board.push(improved_minimax(board, 2, -checkmateVal, checkmateVal, True)[1])
        print(board)
        print()
    else: 
        board.push(improved_minimax(board))
        print(board)
        print()
    # Checks to see if game is over for any reason, ends the while loop and prints the
    # outcome and whose turn it is (False means white checkmated black, and vice versa)
    if(board.is_game_over()):
        gameOver = True
        print(board.outcome().termination)
        print(board.turn)
    # To prevent any cases of games going too long
    elif(board.fullmove_number >= 100):
        gameOver = True
print(board.fullmove_number)