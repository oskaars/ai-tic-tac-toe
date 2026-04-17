# Custom Q-Learning Tic-Tac-Toe AI

A terminal-based Tic-Tac-Toe game featuring a custom-built Reinforcement Learning agent. Instead of using high-level machine learning algorithms, the AI's logic is written entirely from scratch using a Q-Table and the Bellman Equation. The bot learns dynamically in real-time as you play against it.

## Features

* **Real-Time Learning (Online Learning):** The AI evaluates its moves and updates its strategy immediately after every single game.
* **Persistent Memory:** The bot's Q-Table is serialized and saved to a local file (`bot_brain.pkl`). It loads previous knowledge upon startup, allowing training to continue across different sessions.
* **Epsilon-Greedy Strategy:** The bot balances between exploiting its current knowledge and exploring new moves to prevent getting stuck in suboptimal patterns.

## Installation

This project requires Python 3. Install the required dependencies using pip:

```bash
pip install numpy gymnasium
