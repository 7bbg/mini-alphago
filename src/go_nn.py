import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# -----------------------------------
# Go Neural Network (Policy + Value)
# -----------------------------------

class GoNeuralNet(nn.Module):
    def __init__(self, board_size=9):
        super(GoNeuralNet, self).__init__()
        self.board_size = board_size
        self.num_moves = board_size * board_size + 1  # 81 + 1 pass

        # Shared CNN trunk with residual blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.res_block1 = self._build_residual_block(64)
        self.res_block2 = self._build_residual_block(64)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.num_moves)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def _build_residual_block(self, channels):
        """Build a residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        # Shared trunk
        x = F.relu(self.conv1(x)) 
        x = x + self.res_block1(x)
        x = x + self.res_block2(x)

        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

# -----------------------------------
# Board Encoder
# -----------------------------------

def encode_board(board_state, current_player):
    """
    board_state: 2D numpy array (9x9) with:
        - 0: empty
        - 1: black
        - -1: white
    current_player: 1 (black) or -1 (white)

    Returns:
        Tensor shape (3, 9, 9) - channels first (own stones, opponent stones, current player plane)
    """
    own = (board_state == current_player).astype(np.float32)
    opp = (board_state == -current_player).astype(np.float32)
    player_plane = np.full_like(own, 1.0 if current_player == 1 else 0.0)

    encoded = np.stack([own, opp, player_plane]) # shape (3, 9, 9)
    return torch.tensor(encoded, dtype=torch.float32)

# -----------------------------------
# Training Utils
# -----------------------------------

def get_training_setup(model, learning_rate=1e-3):
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return policy_loss_fn, value_loss_fn, optimizer

# -----------------------------------
# Save/Load Helpers
# -----------------------------------

def save_model(model, path="go_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(path="go_model.pth"):
    model = GoNeuralNet()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# -----------------------------------
# Test Routine
# -----------------------------------

if __name__ == "__main__":
    model = GoNeuralNet()
    dummy_board = np.zeros((9, 9), dtype=np.int32)
    current_player = 1 # or -1 for white

    # Encode input
    input_tensor = encode_board(dummy_board, current_player).unsqueeze(0)  # (1, 3, 9, 9)

    # Forward pass
    policy_logits, value = model(input_tensor)

    print("Policy logits shape:", policy_logits.shape)  # [1, 82]
    print("Sample value output:", value.item())          # Scalar in [-1, 1]
    print("Policy logits:", policy_logits)               # Logits for 82 actions (81 moves + pass)
    print("Value output:", value)                        # Value for the board state