import chess
import random

pieceValue = {"K":0, "Q":9, "R":5, "B":3, "N":3, "P":1, "k":0, "q":-9, "r":-5, "b":-3, "n":-3, "p":-1, "None":0}
checkmateVal = 1000
stalemateVal = 0

# Sums the value of all the pieces on the board
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
gameOver = False
board=chess.Board()

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

# Below is code to run a game simulation of two move minimax and random move engines
gameOver = False
board=chess.Board()

while not gameOver:
    if board.turn: 
        board.push(twomove_minimax(board))
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