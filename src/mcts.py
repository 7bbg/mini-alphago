import math
import numpy as np
import torch
import torch.nn.functional as F
from go_game import GoGame, encode_board
from go_nn import GoNeuralNet

class MCTSNode:
    def __init__(self, game, parent=None, prior_prob=1.0):
        """Initialize an MCTS node."""
        self.game = game  # Current game state
        self.parent = parent  # Parent node
        self.children = {}  # Dictionary of child nodes
        self.N = 0  # Visit count
        self.W = 0  # Total value
        self.Q = 0  # Mean value
        self.P = prior_prob  # Prior probability from the policy network

    def is_leaf(self):
        """Check if the node is a leaf (no children)."""
        return len(self.children) == 0

    def expand(self, policy_probs):
        """Expand the node by creating child nodes for all legal moves."""
        legal_moves = self.game.get_legal_moves()
        for move in legal_moves:
            if move not in self.children:
                next_game = self.game.clone()
                next_game.apply_move(move)
                self.children[move] = MCTSNode(next_game, parent=self, prior_prob=policy_probs[move])

    def backup(self, value):
        """Backpropagate the value up the tree."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backup(-value)  # Switch sign for opponent


class MCTS:
    def __init__(self, policy_value_fn, c_puct=1.4, simulations=100):
        """Initialize the MCTS."""
        self.policy_value_fn = policy_value_fn  # Function to get policy and value
        self.c_puct = c_puct  # Exploration constant
        self.simulations = simulations  # Number of simulations to run

    def run(self, root_game):
        """Run MCTS simulations starting from the root game state."""
        root = MCTSNode(game=root_game)

        # Initial expansion and evaluation
        policy_probs, value = self.policy_value_fn(root_game)
        root.expand(policy_probs)

        for _ in range(self.simulations):
            node = root
            # Selection: Traverse the tree to a leaf node
            while not node.is_leaf():
                node = self.select(node)

            # Expansion and evaluation
            if node.game.game_over():
                leaf_value = self.evaluate_winner(node.game)
            else:
                policy_probs, leaf_value = self.policy_value_fn(node.game)
                node.expand(policy_probs)

            # Backpropagation
            node.backup(leaf_value)

        return self.select_action(root)

    def select(self, node):
        """Select the child node with the highest UCB score."""
        best_score = -float('inf')
        best_child = None

        for move, child in node.children.items():
            ucb = child.Q + self.c_puct * child.P * math.sqrt(node.N) / (1 + child.N)
            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child

    def select_action(self, root):
        """Select the best move based on visit counts."""
        move_visits = [(move, child.N) for move, child in root.children.items()]
        moves, visits = zip(*move_visits)
        best_move = moves[np.argmax(visits)]

        # Calculate the output size dynamically (board size squared + 1 for pass move)
        board_size = root.game.size
        output_size = board_size * board_size + 1

        probs = np.zeros(output_size)  # Adjust size for board + pass move
        for move, visit in move_visits:
            probs[move] = visit
        probs = probs / np.sum(probs)
        return best_move, probs

    def evaluate_winner(self, game):
        """Evaluate the winner of the game."""
        black_score = np.sum(game.board == game.BLACK)
        white_score = np.sum(game.board == game.WHITE) + 6.5  # Komi
        result = 1 if black_score > white_score else -1 if black_score < white_score else 0
        return result if game.current_player == game.BLACK else -result


if __name__ == "__main__":
    from go_nn import GoNeuralNet

    # Initialize the neural network model and MCTS
    model = GoNeuralNet()
    def policy_value_fn(game):
        """Get policy and value from the neural network."""
        input_tensor = encode_board(game.board, game.current_player).unsqueeze(0)  # (1, 3, 9, 9)
        with torch.no_grad():
            policy_logits, value_tensor = model(input_tensor)
            policy = torch.softmax(policy_logits, dim=1)[0].numpy()
            value = value_tensor.item()
        return policy, value

    mcts = MCTS(policy_value_fn, simulations=50)

    # Create an empty board and set the starting player
    board = np.zeros((9, 9), dtype=np.int32)
    game = GoGame(board_size=9)
    game.board = board
    game.current_player = 1

    # Run MCTS and get the best move
    best_move, probs = mcts.run(game)
    print("Best move:", best_move)
    print("Move probabilities:", probs)