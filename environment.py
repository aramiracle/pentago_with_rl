import gym
import numpy as np
import torch

class PentagoEnv(gym.Env):
    def __init__(self):
        # Initialize the Pentago board as a 6x6 grid
        self.board = torch.zeros((6, 6), dtype=int)
        self.current_player = 1
        self.winner = None
        self.valid_move_made = False
        self.game_over = False
        self.players_won = set()
        self.max_moves = 36  # Pentago ends after 36 moves
        self.action_space = gym.spaces.Tuple((
            gym.spaces.Discrete(36),  # 36 possible board button actions
            gym.spaces.Discrete(8)    # 8 possible rotation actions
        ))
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(6, 6), dtype=np.float32)
        self.last_row = None
        self.last_col = None

    def reset(self):
        # Reset the Pentago board to its initial state
        self.board = torch.zeros((6, 6), dtype=int)
        self.current_player = 1
        self.winner = None
        self.valid_move_made = False
        self.game_over = False
        self.players_won = set()
        self.last_row = None
        self.last_col = None
        return self.board

    def step(self, action):
        # Action is a tuple (board_button, rotation)
        board_button, rotation = action
        row, col = divmod(board_button, 6)

        # Apply the action to the Pentago environment
        self.board[row, col] = self.current_player
        self.rotate_board_part(rotation // 2, rotation % 2 == 0)  # Adjust rotation values
        self.last_row, self.last_col = row, col

        # Check for a win, draw, or continue the game
        reward, done, info = self.get_reward_done_info()

        # Switch to the next player's turn
        self.current_player = 3 - self.current_player  # Switch between player 1 and player 2

        return self.board, reward, done, info

    def get_reward_done_info(self):
        # Check for a win, draw, or continue the game
        if self.check_win(self.current_player):
            if self.check_win(3 - self.current_player):
                return 50.0, True, {'winner': 'Draw'}
            else:
                return 100.0, True, {'winner': f'Player {self.current_player}'}
        elif self.check_draw():
            return 50.0, True, {'winner': 'Draw'}
        else:
            return -1.0, False, {'winner': 'Game is not finished yet.'}

    def render(self):
        # Display the Pentago board to the console
        print(self.board)

    def check_win(self, player):
        # Check all directions for a win condition
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

    def rotate_board_part(self, corner_index, clockwise=True):
        row_range, col_range = self.get_corner_ranges(corner_index)
        subgrid = self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]]
        rotated_subgrid = np.rot90(subgrid, 3 if clockwise else 1).copy()
        self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]] = torch.tensor(rotated_subgrid, dtype=torch.float32)

    def get_corner_ranges(self, corner_index):
        if corner_index == 0:  # Up-left
            return (0, 3), (0, 3)
        elif corner_index == 1:  # Up-right
            return (0, 3), (3, 6)
        elif corner_index == 2:  # Down-left
            return (3, 6), (0, 3)
        elif corner_index == 3:  # Down-right
            return (3, 6), (3, 6)

    def check_draw(self):
        return np.count_nonzero(self.board) == self.max_moves

    def get_last_move(self):
        return self.last_row, self.last_col
    
    def get_valid_actions(self):
        valid_actions = []

        # Iterate through all possible board_button values
        for board_button in range(36):
            action = (board_button, 0)  # No need to check rotations, set rotation to 0
            if self.is_valid_action(action):
                valid_actions.append(board_button)

        return valid_actions

    def is_valid_action(self, action):
        board_button, _ = action
        row, col = divmod(board_button, 6)

        # Check if the selected cell is empty
        if self.board[row, col] != 0:
            return False

        return True