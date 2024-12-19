# Chess AI with Monte Carlo Tree Search

A Python-based chess AI implementation that uses Monte Carlo Tree Search (MCTS) for move selection. The project includes both a testing framework to evaluate the AI against Stockfish and a graphical interface to play against it.

## Features

- GUI-based chess game interface
- AI opponent using MCTS algorithm
- Testing framework against Stockfish
- Performance metrics and analysis
- Opening book support
- Detailed game analysis and reporting

## Prerequisites

- Python 3.7+
- Stockfish chess engine <a href="https://stockfishchess.org/download/">Download here</a>

## Setup

1. Clone the repository
2. Install the required packages using:

```
pip install chess stockfish pandas matplotlib colorama
```

3. Download Stockfish engine and place `stockfish.exe` in the project root directory
4. Run one of the following commands:

To play against the AI:

```
python test.py
```

To test the AI against Stockfish:

```
python test.py
```

## Project Structure

- `play.py`: GUI and game interface
- `model.py`: Chess AI implementation
- `test.py`: Testing framework
- Generated files:
  - `chess_ai_results.html`: Performance report
  - `win_rate_plot.png`: Visual performance analysis

## How It Works

The chess AI uses Monte Carlo Tree Search (MCTS), a probabilistic search algorithm that balances exploration of new moves with exploitation of known good positions. The AI maintains a tree of possible game states, where each node represents a board position and contains statistics about how successful that position has been in previous simulations. When choosing a move, the AI starts with an opening book of common chess positions, then falls back to MCTS if the position is not in its database.

During each MCTS iteration, the AI selects promising nodes to explore using the UCB1 formula (which balances win rate with exploration), expands the selected node by trying a new move, simulates a game from that position to completion using a mix of random and strategic moves, and finally updates the statistics for all nodes involved in that simulation. This process repeats many times within a specified time limit (default 3 seconds), after which the AI chooses the move that was explored most frequently.

The AI evaluates positions using multiple chess-specific heuristics, including material count, piece positioning, king safety, center control, and pawn structure. It also recognizes tactical patterns like pins, forks, and discovered attacks. These evaluations help guide both the node selection during MCTS and the move choices during simulation, resulting in stronger overall play.

## How it performs

This chess AI went through many iterations, first using a CNN, which very rapidly became too complex for me to train on my PC, I moved away from this approach as I did not want to pay for any server compute. I then did more research and decided that the MCTS algorithm was the best approach for this project. It was a good balance between complexity and performance.

Now while I do know the basics of chess, I am no expert, this made optimizing the model difficult at points as I had to learn chess theory along with it. In its current state, it is a beginner level chess player, but not a strong one. You can play against it with `python play.py`. At the moment it functions purely with classical AI concepts, there is no machine learning involved. I think the best way to improve it would be to implement a learning algorithm to improve the AI's performance over time. I do think that this is a good starting point for a chess AI, and I hope you enjoy playing against it.
