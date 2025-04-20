import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QStatusBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont
from go_game import GoGame
from mcts import MCTS
from go_nn import GoNeuralNet, encode_board
import torch


class GoGUI(QMainWindow):
    def __init__(self, board_size=9):
        super().__init__()
        self.board_size = board_size
        self.game = GoGame(board_size=board_size)

        # Assign roles: Human is Black, AI is White
        self.human_player = self.game.BLACK
        self.ai_player = self.game.WHITE

        # Load AI model and MCTS
        self.model = GoNeuralNet()
        self.model.load_state_dict(torch.load("model_weights.pth"))
        self.model.eval()
        self.mcts = MCTS(self.policy_value_fn, simulations=50)

        self.initUI()

    def initUI(self):
        """Initialize the GUI layout."""
        self.setWindowTitle("Go Game: Human vs AI")
        self.setGeometry(100, 100, 900, 900)

        # Main layout
        main_layout = QVBoxLayout()

        # Board widget
        self.board_widget = BoardWidget(self.game, self)
        main_layout.addWidget(self.board_widget)

        # Control panel
        control_panel = QHBoxLayout()

        # Pass button
        self.pass_button = QPushButton("Pass")
        self.pass_button.setToolTip("Click to pass your turn.")
        self.pass_button.clicked.connect(self.pass_turn)
        control_panel.addWidget(self.pass_button)

        # Restart button
        self.restart_button = QPushButton("Restart")
        self.restart_button.setToolTip("Click to restart the game.")
        self.restart_button.clicked.connect(self.restart_game)
        control_panel.addWidget(self.restart_button)

        # Turn label
        self.turn_label = QLabel("Current Turn: Black (Human)")
        self.turn_label.setAlignment(Qt.AlignCenter)
        self.turn_label.setFont(QFont("Arial", 14))
        control_panel.addWidget(self.turn_label)

        # Player roles label
        self.roles_label = QLabel(f"Human: Black | AI: White")
        self.roles_label.setAlignment(Qt.AlignCenter)
        self.roles_label.setFont(QFont("Arial", 12))
        control_panel.addWidget(self.roles_label)

        main_layout.addLayout(control_panel)

        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Game started. Human plays as Black.")
        self.status_bar.setStyleSheet("background-color: lightgray; font-size: 14px;")

    def policy_value_fn(self, game):
        """Get policy and value from the neural network."""
        input_tensor = encode_board(game.board, game.current_player).unsqueeze(0)  # (1, 3, 9, 9)
        with torch.no_grad():
            policy_logits, value_tensor = self.model(input_tensor)
            policy = torch.softmax(policy_logits, dim=1)[0].numpy()
            value = value_tensor.item()
        return policy, value

    def pass_turn(self):
        """Handle the pass button."""
        print("Pass turn clicked.", self.game.passes)
        self.game.passes += 1
        self.game.switch_player()
        self.update_turn_label()
        self.board_widget.update()

        if self.game.passes >= 2:
            self.end_game()
        else:
            # If it's the AI's turn after the pass, let the AI make a move
            if self.game.current_player == self.ai_player:
                self.status_bar.showMessage("Agent is thinking...")
                self.ai_move()

    def restart_game(self):
        """Restart the game."""
        self.game.reset()
        self.update_turn_label()
        self.board_widget.update()
        self.board_widget.setEnabled(True)  # Re-enable the board
        self.restart_button.setEnabled(True)
        self.pass_button.setEnabled(True)
        self.status_bar.showMessage("Game restarted. Human plays as Black.")

    def update_turn_label(self):
        """Update the turn label."""
        if self.game.current_player == self.human_player:
            current_turn = "Black (Human)" if self.human_player == self.game.BLACK else "White (Human)"
        else:
            current_turn = "Black (AI)" if self.ai_player == self.game.BLACK else "White (AI)"
        self.turn_label.setText(f"Current Turn: {current_turn}")

    def end_game(self):
        """End the game and display the result."""
        black_score, white_score = self.game.score()
        result = "Black wins!" if black_score > white_score else "White wins!" if white_score > black_score else "It's a tie!"
        self.turn_label.setText(result)
        self.board_widget.setEnabled(False)  # Disable the board
        self.board_widget.update()
        self.status_bar.showMessage("Game Over: " + result)

    def ai_move(self):
        """Let the AI make a move."""
        best_move, _ = self.mcts.run(self.game)
        self.game.apply_move(best_move)
        self.update_turn_label()
        self.board_widget.update()

        if self.game.game_over():
            self.end_game()
        else:
            self.status_bar.showMessage("Your turn.")


class BoardWidget(QWidget):
    def __init__(self, game, parent):
        super().__init__(parent)
        self.game = game
        self.parent = parent
        self.setMinimumSize(800, 800)

    def paintEvent(self, event):
        """Draw the board and stones."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # Enable anti-aliasing for smoother rendering
        board_size = self.game.size
        cell_size = self.width() // (board_size + 1)

        # Draw the wooden background
        painter.fillRect(self.rect(), QColor(222, 184, 135))  # Light brown color

        # Draw the grid
        painter.setPen(QPen(Qt.black, 2))
        for i in range(1, board_size + 1):
            painter.drawLine(cell_size * i, cell_size, cell_size * i, cell_size * board_size)
            painter.drawLine(cell_size, cell_size * i, cell_size * board_size, cell_size * i)

        # Draw star points (hoshi) for 9x9 board
        if board_size == 9:
            star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
            for x, y in star_points:
                painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
                painter.drawEllipse((y + 1) * cell_size - 5, (x + 1) * cell_size - 5, 10, 10)

        # Draw the stones at intersections
        for x in range(board_size):
            for y in range(board_size):
                if self.game.board[x, y] == self.game.BLACK:
                    painter.setBrush(QBrush(QColor("black"), Qt.SolidPattern))
                    painter.drawEllipse((y + 1) * cell_size - cell_size // 4, (x + 1) * cell_size - cell_size // 4,
                                        cell_size // 2, cell_size // 2)
                elif self.game.board[x, y] == self.game.WHITE:
                    painter.setBrush(QBrush(QColor("white"), Qt.SolidPattern))
                    painter.drawEllipse((y + 1) * cell_size - cell_size // 4, (x + 1) * cell_size - cell_size // 4,
                                        cell_size // 2, cell_size // 2)

    def mousePressEvent(self, event):
        """Handle mouse clicks to place stones at intersections."""
        if not self.isEnabled():  # Ignore clicks if the board is disabled
            return

        cell_size = self.width() // (self.game.size + 1)
        x = round((event.y() - cell_size) / cell_size)
        y = round((event.x() - cell_size) / cell_size)

        if self.game.in_bounds(x, y) and self.game.is_legal(x, y):
            self.game.apply_move(x * self.game.size + y)
            self.parent.update_turn_label()
            self.update()

            if self.game.game_over():
                self.parent.end_game()
            else:
                # Let the AI make its move
                self.parent.ai_move()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GoGUI(board_size=9)
    gui.show()
    sys.exit(app.exec_())
