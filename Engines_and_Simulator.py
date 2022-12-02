import chess
import random

pieceValue = {"K":0, "Q":9, "R":5, "B":3, "N":3, "P":1,
              "k":0, "q":-9, "r":-5, "b":-3, "n":-3, "p":-1, 
              "None":0}
checkmateVal = 1000
stalemateVal = 0

# piece values based on tables in https://www.chessprogramming.org/Simplified_Evaluation_Function - slighly edited
P_table =       [0,  0,  0,  0,  0,  0,  0,  0,
                .50, .50, .50, .50, .50, .50, .50, .50,
                .10, .10, .20, .30, .30, .20, .10, .10,
                .05,  .05, .10, .25, .25, .10,  .05,  .05,
                0,  0,  0, .20, .20,  0,  0,  0,
                .05, -.05,-.10,  0,  0,-.10, -.05,  .05,
                .05, .10, .10, -.20, -.20, .10, .10,  .05,
                0,  0,  0,  0,  0,  0,  0,  0]

p_table = P_table[::-1]
p_table = [-x for x in p_table]

B_table =       [-.20, -.10, -.10, -.10, -.10, -.10, -.10, -.20,
                -.10, 0,  0,  0,  0,  0,  0, -.10,
                -.10,  0,  .05, .10, .10,  .05,  0, -.10,
                -.10,  .05,  .05, .10, .10,  .05,  .05, -.10,
                -.10,  0, .10, .10, .10, .10,  0, -.10,
                -.10, .10, .10, .10, .10, .10, .10, -.10,
                -.10,  .05,  0,  0,  0,  0,  .05, -.10,
                -.20, -.10, -.10, -.10, -.10, -.10 ,-.10, -.20]

b_table = B_table[::-1]
b_table = [-x for x in b_table]

N_table =       [-.50, -.40, -.30, -.30, -.30, -.30, -.40, -.50,
                -.40, -.20,  0,  0,  0,  0, -.20, -.40,
                -.30,  0, .10, .15, .15, .10,  0, -.30,
                -.30,  .05, .15, .20, .20, .15,  .05, -.30,
                -.30,  0, .15, .20, .20, .15,  0, -.30,
                -.30,  .05, .10, .15, .15, .10,  .05, -.30,
                -.40, -.20,  0,  .05,  .05,  0, -.20, -.40,
                -.50, -.40, -.30, -.30, -.30, -.30, -.40, -.50]

n_table = N_table[::-1] 
n_table = [-x for x in n_table]

R_table =      [0,  0,  0,  0,  0,  0,  0,  0,
               .05, .10, .10, .10, .10, .10, .10, .5,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               -.05,  0,  0,  0,  0,  0,  0, -.05,
               0,  0,  0,  .05,  .05,  0,  0,  0]

r_table = R_table[::-1]
r_table = [-x for x in r_table]

Q_table =       [-.20,-.10,-.10, -.05, -.05,-.10,-.10,-.20,
                -.10,  0,  0,  0,  0,  0,  0,-.10,
                -.10,  0,  .05,  .05,  .05,  .05,  0,-.10,
                 -.05,  0,  .05,  .05,  .05,  .05,  0, -.05,
                  0,  0,  .05,  .05,  .05,  .05,  0, -.05,
                -.10,  .05,  .05,  .05,  .05,  .05,  0,-.10,
                -.10,  0,  .05,  0,  0,  0,  0,-.10,
                -.20,-.10,-.10, -.05, -.05,-.10,-.10,-.20]

q_table = Q_table[::-1]
q_table = [-x for x in q_table]

K_table =       [-.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.30,-.40,-.40,-.50,-.50,-.40,-.40,-.30,
                -.20,-.30,-.30,-.40,-.40,-.30,-.30,-.20,
                -.10,-.20,-.20,-.20,-.20,-.20,-.20,-.10,
                .20, .20,  0,  0,  0,  0, .20, .20,
                .20, .30, .10,  0,  0, .10, .30, .20]
 
k_table = K_table[::-1]
k_table = [-x for x in k_table]
 
pieceTable = {"P": P_table, "p": p_table, "B": B_table, "b": b_table, "N": N_table, "n": n_table,
              "R": R_table, "r": r_table, "Q": Q_table, "q": q_table, "K": K_table, "k": k_table,
              "None": [0] * 64}

# Suming up material using score tables
def improvedScoreBoard(board):
    score = 0
    for square in chess.SQUARES:
        piece = str(board.piece_at(square))
        score = score + pieceValue[piece] + pieceTable[piece][square]
    return score

# This will be used to determine opening, midgame and endgame
def totalScoreBoard(board):
    score = 0
    for square in chess.SQUARES:
        score = score + abs(pieceValue[str(board.piece_at(square))])
    return score

def scoreBoard(board):
    score = 0
    for square in chess.SQUARES:
        score = score + pieceValue[str(board.piece_at(square))]
    return score

def randomMove(board):
    legal_moves = list(board.legal_moves)
    return legal_moves[random. randint(0,len(legal_moves) - 1)]

def firstMove(board):
    return list(board.legal_moves)[0]

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

def greedyMove(board):
    turnVal = 1
    currentScore = 0
    if(not board.turn):
        turnVal = -1;
    maxEval = -checkmateVal
    for move in board.legal_moves:
        board.push(move)
        if(board.is_checkmate()):
            currentScore = checkmateVal
        elif(board.is_stalemate):
            currentScore = stalemateVal
        else: 
            currentScore = turnVal * scoreBoard(board)
        if(currentScore > maxEval):
            maxEval = currentScore
            bestMove = move
        board.pop()
    return bestMove

def twomove_minimax(board):
    turnVal = 1
    if(not board.turn):
        turnVal = -1;
    maxEval = -checkmateVal
    for move in board.legal_moves:
        opp_maxEval = checkmateVal
        bestMove = randomMove(board)
        board.push(move)
        if(board.is_checkmate()):
            board.pop();
            return move;
        for nextmove in board.legal_moves:
            currentScore = 0
            board.push(nextmove)
            if(board.is_checkmate()):
                currentScore = -checkmateVal
            elif(board.is_stalemate):
                currentScore = stalemateVal
            else: 
                currentScore = turnVal * scoreBoard(board)
            if(currentScore < opp_maxEval):
                opp_maxEval = currentScore
            board.pop()
            if(maxEval < opp_maxEval):
                maxEval = opp_maxEval 
                bestMove = move
        board.pop()
    return bestMove

def basic_minimax(board, depth):
    bestmove = None
    if depth == 0 or board.is_game_over():
        return [scoreBoard(board), None]
    if board.turn:
        maxvalue = -checkmateVal
        for move in board.legal_moves:
            board.push(move)
            value = basic_minimax(board, depth-1)[0]
            if(value > maxvalue):
                maxvalue = value
                bestmove = move
            board.pop()
        return [maxvalue, bestmove]
    else:
        minvalue = checkmateVal
        for move in board.legal_moves:
            board.push(move)
            value = basic_minimax(board, depth-1)[0]
            if(value < minvalue):
                minvalue = value
                bestmove = move
            board.pop()
        return [minvalue, bestmove]
    
# same as naiveminimax, but actually looks for checkmate
# next algorithm will use a scoring table to put pieces on optimal squares (better scoring method)
# optimize timing
# Need to return an array, since need to keep track of the current best move as well as the current max/min value
# alpha-beta pruning https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
def improved_minimax(board, depth, alpha, beta):
    bestmove = None
    if depth == 0 or board.is_game_over():
        return [improvedScoreBoard(board), None, None, None]
    if board.turn:
        maxvalue = -checkmateVal
        for move in board.legal_moves:
            board.push(move)
            if(board.is_checkmate()):
                value = checkmateVal
            else: 
                value = improved_minimax(board, depth-1, -beta, -alpha)[0]
            if(value > maxvalue):
                maxvalue = value
                bestmove = move
            board.pop()
            if (maxvalue > alpha):
                alpha = maxvalue
            if (alpha >= beta):
                break
        return [maxvalue, bestmove]
    else:
        minvalue = checkmateVal
        for move in board.legal_moves:
            board.push(move)
            if(board.is_checkmate()):
                value = -checkmateVal
            else:
                value = improved_minimax(board, depth-1, -beta, -alpha)[0]
            if(value < minvalue):
                minvalue = value
                bestmove = move
            board.pop()
            if (minvalue > alpha):
                alpha = minvalue
            if (alpha >= beta):
                break
        return [minvalue, bestmove]

# Below is code to run a game simulation of two move minimax and random move engines
gameOver = False
board=chess.Board()

while not gameOver:
    if board.turn: 
        # White's move, so alpha starts off as lowest score (-checkmateVal) and beta as highest score (checkmateVal)
        # Before alpha beta pruning depth 3 was slow, now able to do depth 4, and depth 5 faster than the previous 
        # depth 4, but still really slow
        board.push(improved_minimax(board, 2, -checkmateVal, checkmateVal)[1])
        print(board)
        print()
    else: 
        board.push(randomMove(board))
        print(board)
        print()
    if(board.is_game_over()):
        gameOver = True
        print(board.outcome().termination)
        print(board.turn)
    elif(board.fullmove_number > 500):
        gameOver = True
print(board.fullmove_number)