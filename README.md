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


```
How to Play
Run the Python script in your terminal.

Upon startup, the script will attempt to load bot_brain.pkl. If the file does not exist, the bot will initialize with zero prior knowledge.

The AI plays as X and you play as O.

Select your move by entering a number from 1 to 9 (corresponding to the board tiles from top-left to bottom-right).

The game runs in an infinite loop. After each match, the bot processes the final reward, updates its internal Q-Table, and immediately resets the board for a new game.

The bot's knowledge is saved automatically (ensure bot.save() is called appropriately in your script loop).

Technologies Used
Python 3 - Core environment and algorithm logic.

Gymnasium - Standard API structure for the Reinforcement Learning environment.

NumPy - Data structures and efficient value calculations.

Pickle - Built-in Python module for state serialization and memory saving.
