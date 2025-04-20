import sys
import os
import unittest
import numpy as np
import torch

# `src` directory 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from go_game import GoGame, encode_board
from go_nn import GoNeuralNet
from self_play import SelfPlay


class TestGoGame(unittest.TestCase):
    def setUp(self):
        self.game = GoGame(board_size=9)

    def test_initial_state(self):
        """Test the initial state of the game."""
        self.assertEqual(self.game.board.shape, (9, 9))
        self.assertTrue(np.all(self.game.board == 0))
        self.assertEqual(self.game.current_player, self.game.BLACK)
        self.assertEqual(self.game.passes, 0)

    def test_switch_player(self):
        """Test switching the current player."""
        self.assertEqual(self.game.current_player, self.game.BLACK)
        self.game.switch_player()
        self.assertEqual(self.game.current_player, self.game.WHITE)

    def test_is_legal(self):
        """Test legal and illegal moves."""
        self.assertTrue(self.game.is_legal(0, 0))  # Empty position
        self.game.board[0, 0] = self.game.BLACK
        self.assertFalse(self.game.is_legal(0, 0))  # Occupied position

    def test_apply_move(self):
        """Test applying a move."""
        self.game.apply_move(0)  # Top-left corner
        self.assertEqual(self.game.board[0, 0], self.game.BLACK)
        self.assertEqual(self.game.current_player, self.game.WHITE)

    def test_game_over(self):
        """Test game over condition."""
        self.assertFalse(self.game.game_over())
        self.game.passes = 2
        self.assertTrue(self.game.game_over())

    def test_score(self):
        """Test scoring."""
        self.game.board[0, 0] = self.game.BLACK
        self.game.board[1, 1] = self.game.WHITE
        black_score, white_score = self.game.score()
        self.assertEqual(black_score, 1)
        self.assertEqual(white_score, 1)


class TestGoNeuralNet(unittest.TestCase):
    def setUp(self):
        self.model = GoNeuralNet(board_size=9)

    def test_forward_pass(self):
        """Test the forward pass of the neural network."""
        dummy_board = np.zeros((9, 9), dtype=np.int32)
        current_player = 1
        input_tensor = encode_board(dummy_board, current_player).unsqueeze(0)  # (1, 3, 9, 9)
        policy_logits, value = self.model(input_tensor)

        self.assertEqual(policy_logits.shape, (1, 82))  # 81 moves + 1 pass
        self.assertEqual(value.shape, (1, 1))
        self.assertTrue(-1 <= value.item() <= 1)  # Value should be in [-1, 1]


class TestSelfPlay(unittest.TestCase):
    def setUp(self):
        self.model = GoNeuralNet(board_size=9)
        self.self_play = SelfPlay(self.model, board_size=9, simulations=2)

    def test_policy_value_fn(self):
        """Test the policy and value function."""
        game = GoGame(board_size=9)
        policy, value = self.self_play.policy_value_fn(game)

        self.assertEqual(policy.shape, (82,))
        self.assertTrue(np.isclose(np.sum(policy), 1))  # Probabilities should sum to 1
        self.assertTrue(-1 <= value <= 1)  # Value should be in [-1, 1]

    def test_play_game(self):
        """Test playing a single self-play game."""
        self.self_play.play_game()
        self.assertGreater(len(self.self_play.data), 0)  # Data should be collected

    def test_save_and_load_data(self):
        """Test saving and loading self-play data."""
        self.self_play.play_game()
        self.self_play.save_data("test_self_play_data")

        # Load the data and verify
        self.self_play.load_data("test_self_play_data")
        self.assertGreater(len(self.self_play.data), 0)


if __name__ == "__main__":
    unittest.main()