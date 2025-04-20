import numpy as np
import copy
import torch

class GoGame:
    def __init__(self, board_size=9):
        self.size = board_size
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = -1

        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.current_player = self.BLACK
        self.history = []  # List of previous board hashes for Ko rule
        self.passes = 0    # Count consecutive passes

    def clone(self):
        """Create a deep copy of the game state."""
        return copy.deepcopy(self)

    def switch_player(self):
        """Switch the current player."""
        self.current_player *= -1

    def get_opponent(self, player):
        """Get the opponent of the given player."""
        return -player

    def encode_board_hash(self):
        """Encode the board state as a hash for Ko rule tracking."""
        return hash(self.board.tobytes())

    def in_bounds(self, x, y):
        """Check if a position is within the board boundaries."""
        return 0 <= x < self.size and 0 <= y < self.size

    def neighbors(self, x, y):
        """Yield all valid neighbors of a position."""
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                yield nx, ny

    def liberties(self, board, x, y, color):
        """Check if a group has at least one liberty."""
        visited = set()
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            visited.add((cx, cy))
            for nx, ny in self.neighbors(cx, cy):
                if board[nx, ny] == self.EMPTY:
                    return True
                elif board[nx, ny] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))
        return False

    def remove_dead_stones(self, board, player):
        """Remove all stones of the opponent that have no liberties."""
        opponent = -player
        to_remove = []
        for x in range(self.size):
            for y in range(self.size):
                if board[x, y] == opponent and not self.liberties(board, x, y, opponent):
                    to_remove.append((x, y))
        for x, y in to_remove:
            board[x, y] = self.EMPTY

    def is_legal(self, x, y):
        """Check if a move is legal."""
        if self.board[x, y] != self.EMPTY:
            return False

        temp_board = self.board.copy()
        temp_board[x, y] = self.current_player
        self.remove_dead_stones(temp_board, self.current_player)

        if not self.liberties(temp_board, x, y, self.current_player):
            return False

        board_hash = hash(temp_board.tobytes())
        if board_hash in self.history:
            return False

        return True

    def get_legal_moves(self):
        """Get all legal moves for the current player."""
        legal = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal(x, y):
                    legal.append(x * self.size + y)
        legal.append(self.size * self.size)  # Pass move
        return legal

    def apply_move(self, move):
        """Apply a move to the board."""
        if move == self.size * self.size:  # Pass move
            self.passes += 1
        else:
            x, y = divmod(move, self.size)
            self.board[x, y] = self.current_player
            self.remove_dead_stones(self.board, self.current_player)

        self.history.append(self.encode_board_hash())
        self.switch_player()

    def game_over(self):
        """Check if the game is over (two consecutive passes)."""
        return self.passes >= 2

    def score(self):
        """Calculate the score for both players. """
        black_score = np.sum(self.board == self.BLACK)
        white_score = np.sum(self.board == self.WHITE)
        return black_score, white_score

    def reset(self):
        """Reset the game to its initial state."""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.current_player = self.BLACK
        self.history.clear()
        self.passes = 0

def encode_board(board, current_player):
    """
    Encode the board state and current player into a tensor format.
    Args:
        board (np.ndarray): The current board state (2D array).
        current_player (int): The current player (1 for BLACK, -1 for WHITE).
    Returns:
        torch.Tensor: Encoded board as a tensor of shape (3, board_size, board_size).
    """
    board_size = board.shape[0]
    black_stones = (board == 1).astype(np.float32)
    white_stones = (board == -1).astype(np.float32)
    current_player_layer = np.full((board_size, board_size), current_player, dtype=np.float32)

    encoded_board = np.stack([black_stones, white_stones, current_player_layer], axis=0)
    return torch.tensor(encoded_board)