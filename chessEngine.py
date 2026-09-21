import pygame
import chess
import torch
import os
import torch.nn as nn
import random
import torch.optim as optim
from flask import Flask, render_template, jsonify, request
import os

#First we are going to make the grid settings of the chess ok ><
#so yeah we can do this project huh><>??
#so i am trying to make web one

depth = 2
epochs = 200
N = 5000
FILE = "chess_model_sucks_lol.pth"

#so lol we are going to get some device details
#i know i am using the "cpu" lol but still i have to do it

device1 = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#printing the device man lol
print(f"device: {device1}")


#make a neural network so that we can do it !
class ChessNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                12,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(
                64 * 8 * 8,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):
        return self.net(x)


model = ChessNet().to(device1)


#load the board lol

def encode(board):

    x = torch.zeros(12, 8, 8)

    for sq, p in board.piece_map().items():

        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)

        channel = p.piece_type - 1

        if p.color == chess.BLACK:
            channel += 6

        x[channel, r, c] = 1

    return x


#give the powers/values to our peices lol

values = {

    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0

}


def score(board):

    return sum(

        len(
            board.pieces(
                p,
                chess.WHITE
            )
        ) * v

        -

        len(
            board.pieces(
                p,
                chess.BLACK
            )
        ) * v

        for p, v in values.items()

    )


def evaluateWork(board):

    if board.is_checkmate():

        return (
            -100000
            if board.turn == chess.WHITE
            else 100000
        )

    if board.is_game_over():

        return 0

    with torch.no_grad():

        x = encode(board).unsqueeze(0).to(device1)

        return model(x).item()


def train():

    print("Training...")

    X = []
    Y = []

    # Make training data

    for i in range(N):

        b = chess.Board()

        for j in range(random.randint(0, 20)):

            if b.is_game_over():
                break

            moves = list(b.legal_moves)

            move = random.choice(moves)

            b.push(move)

        X.append(encode(b))
        Y.append(score(b))


    # Convert to tensors

    X = torch.stack(X).to(device1)

    Y = torch.tensor(
        Y,
        dtype=torch.float32
    ).reshape(-1, 1).to(device1)


    # Training

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_fn = nn.MSELoss()


    for i in range(epochs):

        output = model(X)

        loss = loss_fn(
            output,
            Y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(
            "Epoch:",
            i + 1,
            "Loss:",
            loss.item()
        )


    torch.save(
        model.state_dict(),
        FILE
    )

    print("Model saved.")


if os.path.exists(FILE):

    model.load_state_dict(
        torch.load(
            FILE,
            map_location=device1
        )
    )

    print("Model loaded.")

else:

    train()


model.eval()


def minimax(board, depth, alpha, beta, maximizing):

    #love this algo baby  so yeah learnt that so if i am white and i am attacking i want the max val of
    #my opponent and same goes for my opponet who wants I take the lowest value from them
    #so white is the attacking one and he might be willing to take the queen but black one will think
    #I want white to take my pawn so that i can take the good value from it

    #here depth means the number of times my engine will look forward in future to get the best one
    #board is my board lol
    #alpha is for the min val and maximizing is for the maximum outcome that i can just take from them

    if depth == 0:

        return evaluateWork(board), None


    if maximizing:

        best = -float("inf")
        best_move = None

        for move in board.legal_moves:

            board.push(move)

            value, _ = minimax(
                board,
                depth - 1,
                alpha,
                beta,
                False
            )

            board.pop()

            if value > best:

                best = value
                best_move = move

            alpha = max(
                alpha,
                best
            )

            if beta <= alpha:
                break

        return best, best_move


    else:

        best = float("inf")
        best_move = None

        for move in board.legal_moves:

            board.push(move)

            value, _ = minimax(
                board,
                depth - 1,
                alpha,
                beta,
                True
            )

            board.pop()

            if value < best:

                best = value
                best_move = move

            beta = min(
                beta,
                best
            )

            if beta <= alpha:
                break

        return best, best_move


def ai_move(board):

    value, move = minimax(

        board,

        depth,

        -float("inf"),

        float("inf"),

        False

    )

    print(
        "AI:",
        move,
        "Score:",
        round(value, 2)
    )

    return move


from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

board = chess.Board()


@app.route("/")
def home():
    return send_file("index1.html")


@app.route("/board")
def get_board():

    return jsonify({
        "fen": board.fen(),
        "game_over": board.is_game_over(),
        "result": board.result()
    })


@app.route("/move", methods=["POST"])
def player_move():

    global board

    move = chess.Move.from_uci(
        request.json["move"]
    )

    if move not in board.legal_moves:

        return jsonify({
            "error": "Illegal move"
        }), 400

    board.push(move)

    ai = None

    if not board.is_game_over():

        ai = ai_move(board)

        if ai:
            board.push(ai)

    return jsonify({

        "fen": board.fen(),

        "ai_move": str(ai) if ai else None,

        "game_over": board.is_game_over(),

        "result": board.result()

    })


@app.route("/new", methods=["POST"])
def new_game():

    global board

    board = chess.Board()

    return jsonify({
        "fen": board.fen(),
        "game_over": False
    })


if __name__ == "__main__":
    app.run(debug=True)