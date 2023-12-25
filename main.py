import sys
import torch
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel, QWidget, QVBoxLayout, QGridLayout, QPushButton, QMessageBox, QHBoxLayout, QRadioButton, QInputDialog
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QTimer
import numpy as np
from environment import PentagoEnv
from ddqn import DDQNAgent

class PentagoGame(QMainWindow):
    def __init__(self):
        super().__init__()

        self.rotation_buttons = [
            [QPushButton(), QPushButton()],
            [QPushButton(), QPushButton()],
            [QPushButton(), QPushButton()],
            [QPushButton(), QPushButton()],
        ]

        self.play_game_button = QPushButton("Play Game")
        self.play_again_button = QPushButton("Play Again")

        self.current_player = 1
        self.winner = None
        self.valid_move_made = False
        self.game_over = False
        self.players_won = set()
        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.board = torch.zeros((6, 6), dtype=int)
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

        # Player selection buttons
        player_selection_layout = QHBoxLayout()
        self.first_player_button = QRadioButton("First Player")
        self.second_player_button = QRadioButton("Second Player")
        player_selection_layout.addWidget(self.first_player_button)
        player_selection_layout.addWidget(self.second_player_button)
        layout.addLayout(player_selection_layout)

        # Agent selection button
        agent_selection_button = QPushButton("Select Agent")
        agent_selection_button.clicked.connect(self.select_agent)
        layout.addWidget(agent_selection_button)

        # Play game and Play again buttons
        play_game_layout = QHBoxLayout()
        self.play_game_button.clicked.connect(self.start_game)
        self.play_game_button.setDisabled(True)
        play_game_layout.addWidget(self.play_game_button)

        self.play_again_button.clicked.connect(self.play_again)
        self.play_again_button.setDisabled(True)
        play_game_layout.addWidget(self.play_again_button)

        layout.addLayout(play_game_layout)

        # Status label for game messages
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        self.show()

    def rotate_board_part(self, corner_index, clockwise=True):
        if self.game_over:
            return

        row_range, col_range = self.get_corner_ranges(corner_index)
        subgrid = self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]]
        rotated_subgrid = np.rot90(subgrid, 3 if clockwise else 1)
        rotated_subgrid_tensor = torch.tensor(rotated_subgrid.copy())
        self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]] = rotated_subgrid_tensor
        self.update_board_buttons()

        self.enable_board_buttons()
        self.valid_move_made = False

        self.check_game_over()

        # Switch the turn before checking for win or draw
        self.current_player = 3 - self.current_player

    def get_corner_ranges(self, corner_index):
        if corner_index == 0:
            return (0, 3), (0, 3)
        elif corner_index == 1:
            return (0, 3), (3, 6)
        elif corner_index == 2:
            return (3, 6), (0, 3)
        elif corner_index == 3:
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

            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.toggle_color()
                self.update_board_buttons()

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

    def check_win(self, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for row in range(6):
            for col in range(6):
                if self.board[row, col] == player:
                    for dr, dc in directions:
                        if self.count_aligned(row, col, dr, dc, player) >= 5:
                            return True
        return False
    
    def count_aligned(self, row, col, dr, dc, player):
        count = 1
        count += self.count_direction(row, col, dr, dc, 1, player)
        count += self.count_direction(row, col, dr, dc, -1, player)
        return count

    def count_direction(self, row, col, dr, dc, step, player):
        count = 0
        for i in range(1, 5):
            r, c = row + dr * i * step, col + dc * i * step
            if 0 <= r < 6 and 0 <= c < 6 and self.board[r, c] == player:
                count += 1
            else:
                break
        return count

    def check_draw(self):
        return all(self.board[row, col] != 0 for row in range(6) for col in range(6))

    def start_game(self):
        self.game_over = False
        self.players_won = set()
        self.status_label.setText("")
        self.current_player = 1 if self.first_player_button.isChecked() else 2
        self.enable_board_buttons()

        if self.current_player == 2:
            self.play_ai_turn()

    def play_again(self):
        self.game_over = False
        self.players_won = set()
        self.status_label.setText("")
        self.enable_board_buttons()
        self.current_player = 1 if self.first_player_button.isChecked() else 2

        if self.current_player == 2:
            self.play_ai_turn()

    def select_agent(self):
        agent_type, ok = QInputDialog.getItem(self, "Select Agent Type", 
                                            "Choose an agent:", ["DDQN"], 0, False)
        if ok and agent_type:
            if agent_type == "DDQN":
                if self.current_player == 1:
                    self.agent = DDQNAgent(PentagoEnv())
                    self.load_agent('saved_agents/ddqn_agents_after_train.pth', player=2)
                else:
                    self.agent = DDQNAgent(PentagoEnv())
                    self.load_agent('saved_agents/ddqn_agents_after_train.pth', player=1)

    def load_agent(self, filepath, player):
        try:
            # Load the agent based on its type and player
            if isinstance(self.agent, DDQNAgent):
                # Load DQN agent
                checkpoint = torch.load(filepath)
                if player == 1:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1'])
                    self.agent.target_model.load_state_dict(checkpoint['target_model_state_dict_player1'])
                elif player == 2:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2'])
                    self.agent.target_model.load_state_dict(checkpoint['target_model_state_dict_player2'])

            # Display a success message
            self.status_label.setText(f"{type(self.agent).__name__} loaded successfully for Player {player}.")
            self.play_game_button.setDisabled(False)
            self.play_again_button.setDisabled(False)
        except FileNotFoundError:
            # Display an error message if the file is not found
            self.status_label.setText("Agent file not found.")
        except Exception as e:
            # Display an error message if loading fails
            self.status_label.setText(f"Failed to load agent: {str(e)}")

    def play_ai_turn(self):
        if self.game_over:
            return

        # Disable board buttons and rotation buttons during AI's turn
        self.disable_board_buttons()
        self.disable_rotation_buttons()

        action = self.ai_select_move()
        row, col, rotation, clockwise = self.agent.env.action_to_move(action)

        # Update the game state with the selected move
        self.board[row, col] = self.current_player
        self.rotate_board_part(rotation, clockwise)

        # Update the board buttons and toggle the color
        self.update_board_buttons()
        self.toggle_color()

        # Update the UI to highlight the AI's move
        self.update_ui(action)

        # Check for a win or draw after the AI move
        self.check_game_over()

        # Enable board buttons and rotation buttons after AI's turn
        self.enable_board_buttons()
        self.enable_rotation_buttons()


    def disable_rotation_buttons(self):
        for i in range(4):
            for j in range(2):
                self.rotation_buttons[i][j].setDisabled(True)

    def enable_rotation_buttons(self):
        for i in range(4):
            for j in range(2):
                self.rotation_buttons[i][j].setEnabled(True)

    def ai_select_move(self):
        if  isinstance(self.agent, DDQNAgent):
            return self.agent.select_action(self.agent.env.board, epsilon=0)
        else:
            self.status_label.setText("No agent is loaded.")
            return None
        
    def update_game_state(self, action):
        # Convert the flattened action index to the corresponding (row, col, rotation) values
        row, col, rotation, clockwise = self.agent.env.action_to_move(action)

        # Update the game state with the selected move
        self.board[row, col] = self.current_player
        self.rotate_board_part(rotation, clockwise)

        # Update the board buttons and toggle the color
        self.update_board_buttons()
        self.toggle_color()

    def update_ui(self, action):
        row, col, rotation, clockwise = self.agent.env.action_to_move(action)
        
        # Change the color of the selected button
        button = self.board_buttons[row][col]
        button.setStyleSheet(f"background-color: {self.current_color.name()}; "
                            "border: 1px solid black; border-radius: 30px; color: black;")

        # Update the UI to highlight the AI's move with a delay
        QTimer.singleShot(1000, lambda: self.delayed_rotate_board(rotation, clockwise))

    def delayed_rotate_board(self, rotation, clockwise):
        # Rotate the board part after the delay
        self.rotate_board_part(rotation, clockwise)

        # Update the board buttons and toggle the color
        self.update_board_buttons()

    def check_game_over(self):
        # Check for a win or draw after the current move
        if self.check_win(self.current_player):
            if self.check_win(3 - self.current_player):
                print("It's a draw!")
                self.game_over = True
                self.show_result()
            else:
                print(f"Player {self.current_player} wins!")
                self.winner = self.current_player
                self.game_over = True
                self.show_result()
        elif self.check_draw():
            print("It's a draw!")
            self.game_over = True
            self.show_result()
            
    def show_result(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Game Over")

        if self.winner:
            winner = f"Player {self.winner}"
            msg_box.setText(f"{winner} wins!")
        else:
            msg_box.setText("It's a draw!")

        msg_box.exec()

def main():
    app = QApplication(sys.argv)
    game = PentagoGame()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
