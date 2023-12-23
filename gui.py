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
                button.setFixedSize(60, 60)  # Larger button size
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
                rotate_button.setFixedSize(30, 30)  # Larger button size
                rotate_button.setStyleSheet("font-size: 18px; font-weight: bold;")

                # Set text based on direction
                rotate_button.setText("↻" if j == 0 else "↺")

                # Connect the button click to the rotation function
                rotate_button.clicked.connect(lambda _, idx=i, clockwise=j == 0: self.rotate_board_part(idx, clockwise=clockwise))

                rotate_layout.addWidget(rotate_button, i // 2 * 3 + 1, (i % 2) * 3 + j * 2)

        layout.addLayout(rotate_layout)
        layout.addStretch()


        for i in range(4):
            self.rotation_buttons[i][0].clicked.connect(lambda _, idx=i: self.rotate_board_part(idx, clockwise=True))
            self.rotation_buttons[i][1].clicked.connect(lambda _, idx=i: self.rotate_board_part(idx, clockwise=False))

        self.current_color = QColor('blue')
        self.setWindowTitle('Pentago Game')
        self.setGeometry(100, 100, 400, 400)  # Adjusted window size
        self.show()


    def rotate_board_part(self, corner_index, clockwise=True):
        row_range, col_range = self.get_corner_ranges(corner_index)
        subgrid = self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]]
        rotated_subgrid = np.rot90(subgrid, 3 if clockwise else 1)
        self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]] = rotated_subgrid
        self.update_board_buttons()

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
        sender_button = self.sender()

        if sender_button:
            row, col = self.get_button_position(sender_button)
            current_color = 'blue' if self.current_color == QColor('blue') else 'red'
            
            # Check if the button is not already colored
            if sender_button.styleSheet().find(current_color) == -1:
                self.toggle_color()
                self.board[row, col] = 1 if self.current_color == QColor('blue') else 2
                self.update_board_buttons()

    def toggle_color(self):
        self.current_color = QColor('blue') if self.current_color == QColor('red') else QColor('red')

    def get_button_position(self, button):
        for i in range(6):
            for j in range(6):
                if self.board_buttons[i][j] == button:
                    return i, j

def main():
    app = QApplication(sys.argv)
    game = PentagoGame()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
