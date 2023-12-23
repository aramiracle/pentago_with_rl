import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton
from PyQt6.QtGui import QColor
import numpy as np

class PentagoGame(QWidget):
    def __init__(self):
        super().__init__()

        # Create the rotation buttons for each corner
        self.rotation_buttons = [
            [QPushButton(), QPushButton()],
            [QPushButton(), QPushButton()],
            [QPushButton(), QPushButton()],
            [QPushButton(), QPushButton()],
        ]

        self.current_player = 1  # Player 1 starts
        self.valid_move_made = False
        self.game_over = False  # Track whether the game is over
        self.players_won = set()  # Track players who have won
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Create the game board grid
        self.board = np.zeros((6, 6), dtype=int)
        self.board_buttons = [[QPushButton() for _ in range(6)] for _ in range(6)]

        board_layout = QGridLayout()

        for row in range(6):
            for col in range(6):
                button = QPushButton()
                button.setFixedSize(60, 60)
                button.setStyleSheet(
                    "QPushButton { background-color: #e0e0e0; border: 1px solid black; border-radius: 30px; color: black; }"
                    "QPushButton:hover { background-color: #c0c0c0; }"
                )
                button.clicked.connect(self.board_button_clicked)
                self.board_buttons[row][col] = button
                board_layout.addWidget(button, row, col)

        layout.addLayout(board_layout)

        # Create rotation buttons for each corner
        rotate_layout = QGridLayout()

        for i in range(4):
            for j in range(2):
                rotate_button = QPushButton()
                rotate_button.setFixedSize(30, 30)
                rotate_button.setStyleSheet("font-size: 18px; font-weight: bold;")
                rotate_button.setText("↻" if j == 0 else "↺")
                rotate_button.clicked.connect(lambda _, idx=i, clockwise=j == 0: self.rotate_board_part(idx, clockwise=clockwise))
                rotate_layout.addWidget(rotate_button, i // 2 * 3 + 1, (i % 2) * 3 + j * 2)

        layout.addLayout(rotate_layout)
        layout.addStretch()

        for i in range(4):
            self.rotation_buttons[i][0].clicked.connect(lambda _, idx=i: self.rotate_board_part(idx, clockwise=True))
            self.rotation_buttons[i][1].clicked.connect(lambda _, idx=i: self.rotate_board_part(idx, clockwise=False))

        self.current_color = QColor('blue')
        self.setWindowTitle('Pentago Game')
        self.setGeometry(100, 100, 400, 400)
        self.show()

    def rotate_board_part(self, corner_index, clockwise=True):
        if self.game_over:
            return

        row_range, col_range = self.get_corner_ranges(corner_index)
        subgrid = self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]]
        rotated_subgrid = np.rot90(subgrid, 3 if clockwise else 1)
        self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]] = rotated_subgrid
        self.update_board_buttons()

        # Enable board buttons after rotation
        self.enable_board_buttons()
        self.valid_move_made = False  # Reset the flag for the next player's move

        # Switch to the next player's turn
        self.current_player = 3 - self.current_player  # Switch between player 1 and player 2

        # Check for a win after each rotation
        if self.check_win():
            self.players_won.add(self.current_player)
            if len(self.players_won) == 2:
                print("It's a draw!")
                self.game_over = True
                # You can perform any additional actions here for a draw condition
            else:
                print(f"Player {self.current_player} wins!")
                self.game_over = True
                # You can perform any additional actions here for a win condition
        elif self.check_draw():
            print("It's a draw!")
            self.game_over = True
            # You can perform any additional actions here for a draw condition

    def get_corner_ranges(self, corner_index):
        if corner_index == 0:  # Up-left
            return (0, 3), (0, 3)
        elif corner_index == 1:  # Up-right
            return (0, 3), (3, 6)
        elif corner_index == 2:  # Down-left
            return (3, 6), (0, 3)
        elif corner_index == 3:  # Down-right
            return (3, 6), (3, 6)

    def update_board_buttons(self):
        for row in range(6):
            for col in range(6):
                button = self.board_buttons[row][col]
                value = self.board[row, col]
                color = 'blue' if value == 1 else 'red' if value == 2 else '#e0e0e0'
                button.setStyleSheet(
                    f"background-color: {color}; "
                    "border: 1px solid black; border-radius: 30px; color: black;"
                )

    def board_button_clicked(self):
        if self.game_over:
            return

        sender_button = self.sender()

        if sender_button and not self.valid_move_made:
            row, col = self.get_button_position(sender_button)

            # Check if the selected button is not already occupied
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.toggle_color()
                self.update_board_buttons()

                # Disable board buttons until rotation
                self.disable_board_buttons()
                self.valid_move_made = True

    def toggle_color(self):
        self.current_color = QColor('blue') if self.current_color == QColor('red') else QColor('red')

    def get_button_position(self, button):
        for i in range(6):
            for j in range(6):
                if self.board_buttons[i][j] == button:
                    return i, j

    def disable_board_buttons(self):
        for row in range(6):
            for col in range(6):
                self.board_buttons[row][col].setDisabled(True)

    def enable_board_buttons(self):
        for row in range(6):
            for col in range(6):
                self.board_buttons[row][col].setEnabled(True)

    def check_win(self):
        # Check all directions for a win condition
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for row in range(6):
            for col in range(6):
                for dr, dc in directions:
                    if self.count_aligned(row, col, dr, dc) >= 5:
                        return True
        return False

    def count_aligned(self, row, col, dr, dc):
        count = 1
        count += self.count_direction(row, col, dr, dc, 1)
        count += self.count_direction(row, col, dr, dc, -1)
        return count

    def count_direction(self, row, col, dr, dc, step):
        count = 0
        for i in range(1, 5):  # Adjusted the range to 5 for a total of 5 pieces
            r, c = row + dr * i * step, col + dc * i * step
            if 0 <= r < 6 and 0 <= c < 6 and self.board[r, c] == self.current_player:
                count += 1
            else:
                break
        return count

    def check_draw(self):
        return all(self.board[row, col] != 0 for row in range(6) for col in range(6))

def main():
    app = QApplication(sys.argv)
    game = PentagoGame()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
