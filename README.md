# Mini AlphaGo

Mini AlphaGo is a simplified implementation of the AlphaGo algorithm for the game of Go. It includes a Go game engine, a neural network for policy and value prediction, Monte Carlo Tree Search (MCTS) for decision-making, and a self-play training loop. The project also features a PyQt-based graphical user interface (GUI) for playing against the Go-Playing agent.

---

## Features

1. **Go Game Engine**:
   - Supports a 9x9 Go board.
   - Implements basic rules of Go, including captures, Ko rule, and scoring.
   - Detects game-over conditions (two consecutive passes).

2. **Neural Network**:
   - A PyTorch-based neural network with shared convolutional layers and separate policy and value heads.
   - Encodes the board state and current player into a tensor format.
   - Outputs move probabilities (policy) and win probability (value).

3. **Monte Carlo Tree Search (MCTS)**:
   - Guides move exploration using the policy network.
   - Evaluates leaf nodes using the value network.
   - Backpropagates values to improve decision-making.

4. **Self-Play and Training**:
   - Plays games between Go-Playing agents to generate training data.
   - Trains the neural network using the collected data.
   - Saves and loads model weights for continuous improvement.

5. **Graphical User Interface (GUI)**:
   - A PyQt-based GUI for playing against the Go-Playing agent.
   - Displays the board, stones, and current turn.
   - Includes buttons for passing, restarting, and displaying game results.

---

## Installation

### Prerequisites
- Python 3.8 or later
- PyTorch
- PyQt5
- NumPy

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mini-alphago.git
   cd mini-alphago 
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the installation by running the test suite:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```
---
## Usage
1. ####  Play Against the Agent
    Run the GUI to play against the agent:
    ```bash
    python src/go_gui.py
    ```

#### Features:
- **Pass Button**: Pass your turn.
- **Restart Button**: Restart the game.
- **Game Over Detection**: The game ends after two consecutive passes, and the winner is displayed.

2. #### Train the Go-Playing agent
Run the self-play script to train the Go-Playing agent:
```bash
    python src/self_play.py
```

#### Steps:
1. The agent plays multiple self-play games.
2. Training data (state, policy, value) is saved to disk.
3. The neural network is trained using the collected data.
4. The trained model weights are saved for future use.

---

3. ####  Test the Code
Run the test suite to ensure everything is working correctly:

```bash
    python -m unittest discover -s tests -p "test_*.py"
```

---



## Project Structure

```markdown
mini-alphago/
├── src/
│   ├── go_game.py          # Go game engine
│   ├── go_nn.py            # Neural network (policy + value)
│   ├── mcts.py             # Monte Carlo Tree Search
│   ├── self_play.py        # Self-play and training loop
│   ├── go_gui.py           # PyQt-based GUI
├── tests/
│   ├── test_mini_alphago.py # Unit tests for the project
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
### Future Improvements
1. **Support for Larger Boards**:

    - Extend the implementation to support `13x13` and `19x19` boards.
2. **Enhanced network**:
    - Residual networks.
    - Train using reinforcement learning, incorporating discounted rewards or more refined methods for  credit assignment.
3. **Improved GUI**:

    - Add a timer for each player's turn.
    - Allow users to choose their role (Black or White).
4. **Performance Optimization**:
    - Optimize MCTS for faster decision-making.
    - Use GPU acceleration for training and inference.


### Acknowledgments
This project is inspired by DeepMind's AlphaGo and serves as an educational implementation of its core concept