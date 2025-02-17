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
        self.previous_board = None # Store board state before the current move

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
        self.previous_board = None
        return self.board

    def step(self, action):
        # Action is a tuple (board_button, rotation)
        board_button, rotation = action
        row, col = divmod(board_button, 6)

        self.previous_board = self.board.clone() # Store the board before current move

        # Place the piece
        if self.board[row, col] != 0:
            raise ValueError("Invalid move: Cell is already occupied")
        self.board[row, col] = self.current_player
        self.last_row, self.last_col = row, col

        # Rotate the board part
        self.rotate_board_part(rotation // 2, rotation % 2 == 0)  # Adjust rotation values

        # Check for game end and calculate reward
        reward, done, info = self.get_reward_done_info(action) # Pass action to reward function

        # Switch player if game is not over
        if not done:
            self.current_player = 3 - self.current_player  # Switch between player 1 and player 2

        return self.board, reward, done, info

    def get_reward_done_info(self, action):
        # Calculate immediate reward based on placed piece
        progress_reward = self.calculate_progress_reward(self.current_player, self.last_row, self.last_col)

        block_opponent_win_reward = self.check_block_opponent_win() * 60 # Reward for blocking opponent win (increased to 60)
        create_winning_threat_reward = self.check_create_winning_threat() * 20 # Reward for creating winning threat
        give_opponent_winning_move_penalty = self.check_give_opponent_winning_move() * -80 # Penalty for giving opponent winning move (increased to -80)


        # Check for win/draw
        win_player = None
        if self.check_win(1):
            win_player = 1
        if self.check_win(2):
            if win_player is not None: # Both players won in the same move, which is draw in Pentago
                return 50.0, True, {'winner': 'Draw'}
            win_player = 2

        if win_player is not None:
            if win_player == self.current_player:
                return 100.0 + progress_reward + block_opponent_win_reward + create_winning_threat_reward + give_opponent_winning_move_penalty, True, {'winner': f'Player {self.current_player}'}
            else:
                return -100.0 + give_opponent_winning_move_penalty, True, {'winner': f'Player {win_player}'} # Opponent won, penalize

        if self.check_draw():
            return 50.0, True, {'winner': 'Draw'}

        # Game continues, give a small negative reward to encourage faster win, plus other rewards/penalties
        return -1 + progress_reward + block_opponent_win_reward + create_winning_threat_reward + give_opponent_winning_move_penalty, False, {'winner': 'Game is not finished yet.'}


    def calculate_progress_reward(self, player, row, col):
        max_aligned = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            aligned_count = self.count_aligned(row, col, dr, dc, player)
            max_aligned = max(max_aligned, aligned_count)

        # Reward scaled to the progress towards winning (5 in a row)
        return (max_aligned - 1) * 5  # reward from 0 to 20 for 1 to 5 aligned


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
        valid_actions_list = []
        for board_button in range(36):
            if self.board[divmod(board_button, 6)] == 0: # check if cell is empty
                valid_actions_list.append(board_button)
        return valid_actions_list

    def is_valid_action(self, action):
        board_button, _ = action
        row, col = divmod(board_button, 6)
        return self.board[row, col] == 0

    def action_to_move(self, action):
        # Convert the flattened action index to the corresponding (row, col, rotation) values
        board_button, rotation = action
        row, col = divmod(board_button, 6)
        return row, col, rotation // 2, rotation % 2 == 0

    def clone(self):
        new_env = PentagoEnv()
        new_env.board = self.board.clone()
        new_env.current_player = self.current_player
        new_env.winner = self.winner
        new_env.last_row = self.last_row
        new_env.last_col = self.last_col
        return new_env

    def check_block_opponent_win(self):
        opponent_player = 3 - self.current_player
        opponent_winning_moves_before = self._get_winning_moves(opponent_player, self.previous_board) # Winning moves opponent *had* before current move
        opponent_winning_moves_after = self._get_winning_moves(opponent_player, self.board) # Winning moves opponent has *now*

        if opponent_winning_moves_before and not opponent_winning_moves_after:
            return 1 # Blocked opponent win
        return 0

    def check_create_winning_threat(self):
        current_player_winning_moves = self._get_winning_moves(self.current_player, self.board)
        if current_player_winning_moves:
            return 1 # Created winning threat (or immediate win)
        return 0

    def check_give_opponent_winning_move(self):
        opponent_player = 3 - self.current_player
        opponent_winning_moves = self._get_winning_moves(opponent_player, self.board)
        if opponent_winning_moves:
            return 1 # Gave opponent winning move
        return 0

    def _get_winning_moves(self, player, current_board):
        winning_moves = []
        for board_button in range(36):
            row, col = divmod(board_button, 6)
            if current_board[row, col] == 0:
                for rotation in range(8):
                    temp_board = current_board.clone()
                    temp_board[row, col] = player
                    temp_env = self.clone() # Use clone to avoid modifying original env state
                    temp_env.board = temp_board
                    temp_env.rotate_board_part(rotation // 2, rotation % 2 == 0)
                    if temp_env.check_win(player):
                        winning_moves.append(((board_button, rotation)))
        return winning_moves
