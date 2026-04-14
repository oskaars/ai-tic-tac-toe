# Tic-Tac-Toe Reinforcement Learning

A terminal-based Tic-Tac-Toe game built to demonstrate Reinforcement Learning concepts. You can play against a pre-trained AI model or train your own from scratch.

## Installation

This project requires Python 3. Install the required dependencies using pip:

```bash
pip install gymnasium stable-baselines3 numpy
```

## How to Play

By default, the script is configured for gameplay against the existing AI model.

1. Ensure the `game()` function is called at the very bottom of the file.
2. Run the script in your terminal.
3. The AI plays as **X** and you play as **O**. 
4. When prompted, select your move by entering a number from `1` to `9` (corresponding to the board tiles from top-left to bottom-right).

## Training a New Model

You can train a new AI model by making it play 100,000 games against a random-move bot. 

To enable training mode:

1. Inside the `step()` method, uncomment the line responsible for the opponent's random moves (around line 61):
   ```python
   self.makeMove(random.choice(self.availableMoves()), "O")
   ```
2. At the bottom of the file, remove or comment out the `game()` function call, and add/uncomment the `training()` function call.
3. Run the script. The training process will start, and the new model will be saved to your disk as `tic_tac_toe_ai.zip`.

**Important:** Once the training is complete, remember to revert the changes. Comment out the `random.choice` line in the `step()` method and switch the function call at the bottom back to `game()` before trying to play.

## Technologies Used

* **Python 3** - Core game logic.
* **Gymnasium** - Standard API for Reinforcement Learning environments.
* **Stable-Baselines3 (PPO)** - Implementation of the Proximal Policy Optimization algorithm.
* **NumPy** - Matrix operations for converting the board state into an AI-readable format.
