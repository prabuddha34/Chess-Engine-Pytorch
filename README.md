# ♟️ Chess Engine — PyTorch

A chess engine built from scratch using **Python, PyTorch, Minimax, Alpha-Beta Pruning, and python-chess**.

This project started as me messing around with chess, neural networks, and game algorithms — and slowly turned into an actual playable web chess engine.

## 🚀 Features

* ♟️ Chess board and legal move handling
* 🧠 Minimax algorithm
* ✂️ Alpha-Beta pruning
* 🤖 PyTorch-based evaluation model
* 📊 Material-based position evaluation
* 🌐 Flask backend
* 💻 Web-based interface
* 🔥 Runs on CPU or CUDA when available
* 📦 Deployable to Railway

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **python-chess**
* **Flask**
* **Gunicorn**
* **HTML / CSS / JavaScript**

## 🧠 How the Engine Works

The engine represents a chess position as a tensor and evaluates the position using a PyTorch neural network.

For searching moves, it uses **Minimax** with **Alpha-Beta pruning**.

Basically:

```text
Current Position
       ↓
Generate Legal Moves
       ↓
Try Possible Moves
       ↓
Minimax Search
       ↓
Alpha-Beta Pruning
       ↓
Evaluate Position
       ↓
Choose Move
```

The current evaluation function is still relatively simple, so this is **not meant to compete with engines like Stockfish**.

The main goal of this project is learning how chess engines, search algorithms, neural networks, and web deployment work together.

## 📁 Project Structure

```text
Chess-Engine-Pytorch/
│
├── chessEngine.py
├── index.html
├── requirements.txt
├── chess_model_sucks_lol.pth
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/prabuddha34/Chess-Engine-Pytorch.git
```

Enter the project:

```bash
cd Chess-Engine-Pytorch
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

Start the Flask application:

```bash
python chessEngine.py
```

Or run it with Gunicorn:

```bash
gunicorn chessEngine:app
```

Then open the local address shown by Flask in your browser.

## 🌐 Deployment

The project can be deployed as a Flask web application.

The production server uses:

```bash
gunicorn chessEngine:app
```

The application can be hosted on services such as **Railway**.

## 📦 Requirements

```text
flask
gunicorn
torch
python-chess
```

> Pygame is not required for the web deployment. It was used during the earlier desktop version of the project.

## 🔮 Future Improvements

There is still a LOT I want to improve.

* Better neural-network training
* Better chess evaluation
* Opening book
* Piece-square tables
* Transposition tables
* Move ordering
* Stronger minimax search
* Iterative deepening
* Better web UI
* Human vs AI mode
* AI vs AI mode
* Chess clocks
* Game history
* Checkmate / draw UI
* Stockfish comparison

## 🎯 Why I Built This

I wanted to understand what actually happens inside a chess engine instead of just using one.

This project lets me experiment with:

* Game-tree search
* Minimax
* Alpha-Beta pruning
* Neural networks
* PyTorch
* Chess representations
* Flask APIs
* Web deployment

It's basically a learning project that got way more interesting than I expected.

## 👨‍💻 Author

**Prabuddha Pal**

GitHub: [@prabuddha34](https://github.com/prabuddha34)

---

⭐ If you find the project interesting, feel free to star the repository.
