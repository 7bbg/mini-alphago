import numpy as np
import torch
from go_game import GoGame
from mcts import MCTS
from go_nn import GoNeuralNet, encode_board

class SelfPlay:
    def __init__(self, model, board_size=9, simulations=50):
        """Initialize the self-play environment."""
        self.model = model
        self.board_size = board_size
        self.simulations = simulations
        self.mcts = MCTS(self.policy_value_fn, simulations=simulations)
        self.data = []  # To store training data (state, policy, value)

    def policy_value_fn(self, game):
        """Get policy and value from the neural network."""
        input_tensor = encode_board(game.board, game.current_player).unsqueeze(0)  # (1, 3, 9, 9)
        with torch.no_grad():
            policy_logits, value_tensor = self.model(input_tensor)
            policy = torch.softmax(policy_logits, dim=1)[0].numpy()
            value = value_tensor.item()
        return policy, value

    def play_game(self):
        """Play a single self-play game."""
        game = GoGame(board_size=self.board_size)
        game.reset()
        game_data = []

        while not game.game_over():
            best_move, probs = self.mcts.run(game)
            state = encode_board(game.board, game.current_player).numpy()
            game_data.append((state, probs, None))
            game.apply_move(best_move)

        black_score, white_score = game.score()
        result = 1 if black_score > white_score else -1 if black_score < white_score else 0

        for i in range(len(game_data)):
            game_data[i] = (game_data[i][0], game_data[i][1], result if i % 2 == 0 else -result)

        self.data.extend(game_data)

    def save_data(self, filename_prefix):
        """Save the collected data to separate files."""
        states, policies, values = zip(*self.data)
        np.save(f"{filename_prefix}_states.npy", np.array(states))
        np.save(f"{filename_prefix}_policies.npy", np.array(policies))
        np.save(f"{filename_prefix}_values.npy", np.array(values))

    def load_data(self, filename_prefix):
        """Load the collected data from separate files."""
        states = np.load(f"{filename_prefix}_states.npy", allow_pickle=True)
        policies = np.load(f"{filename_prefix}_policies.npy", allow_pickle=True)
        values = np.load(f"{filename_prefix}_values.npy", allow_pickle=True)
        self.data = list(zip(states, policies, values))

    def train(self, epochs=10, batch_size=32, learning_rate=0.001):
        """Train the neural network using the collected data."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn_policy = torch.nn.CrossEntropyLoss()
        loss_fn_value = torch.nn.MSELoss()

        for epoch in range(epochs):
            np.random.shuffle(self.data)
            for i in range(0, len(self.data), batch_size):
                batch = self.data[i:i + batch_size]
                states, policies, values = zip(*batch)

                # Convert to tensors
                states = torch.tensor(np.array(states), dtype=torch.float32)
                policies = torch.tensor(np.array(policies), dtype=torch.float32)
                values = torch.tensor(np.array(values), dtype=torch.float32)

                # Forward pass
                policy_logits, value_preds = self.model(states)
                policy_loss = loss_fn_policy(policy_logits, policies)
                value_loss = loss_fn_value(value_preds.squeeze(), values)
                loss = policy_loss + value_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def train_with_reinforcement_learning(self, epochs=10, learning_rate=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.play_game()  # Generate self-play data
            rewards = []  # Store rewards for each move

            for state, policy, result in self.data:
                # Calculate the reward (e.g., game result)
                rewards.append(result)

            # Convert rewards to a tensor
            rewards = torch.tensor(rewards, dtype=torch.float32)

            # Forward pass
            states = torch.tensor([d[0] for d in self.data], dtype=torch.float32)
            policies = torch.tensor([d[1] for d in self.data], dtype=torch.float32)
            policy_logits, values = self.model(states)

            # Policy loss (using policy gradient)
            policy_loss = -torch.sum(policies * torch.log_softmax(policy_logits, dim=1) * rewards)

            # Value loss
            value_loss = torch.nn.functional.mse_loss(values.squeeze(), rewards)

            # Total loss
            loss = policy_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    # Initialize the neural network model
    model = GoNeuralNet()

    # Load pre-trained weights if available
    try:
        model.load_state_dict(torch.load("model__weights.pth"))
        print("Loaded pre-trained model weights.")
    except FileNotFoundError:
        print("No pre-trained weights found. Starting from scratch.")

    # Initialize self-play environment
    self_play = SelfPlay(model, board_size=9, simulations=2) # Adjust simulations as needed

    # Play multiple self-play games
    for _ in range(10):  # Play 100 games
        print("Playing game...")
        self_play.play_game()

    # Save the collected data
    self_play.save_data("self_play_data")

    # Train the model
    self_play.train(epochs=10, batch_size=32, learning_rate=0.001)

    # Save the trained model weights
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model training complete and weights saved.")