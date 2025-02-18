import gym
import numpy as np
import torch
import timeit

class PentagoEnv2(gym.Env):
    def __init__(self):
        # Initialize the Pentago board as a 6x6 grid
        self.board = torch.zeros((6, 6), dtype=int)
        self.current_player = 1
        self.winner = None
        self.valid_move_made = False
        self.game_over = False
        self.players_won = set()
        self.max_moves = 36  # Pentago ends after 36 moves
        self.action_space = gym.spaces.Discrete(36 * 8)  # Combine board_button and rotation
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
        # Action is a single integer representing the flattened (board_button, rotation) pair
        board_button, rotation = divmod(action, 8)
        row, col = divmod(board_button, 6)

        self.previous_board = self.board.clone() # Store the board before current move

        # Place the piece
        if self.board[row, col] != 0:
            raise ValueError("Invalid move: Cell is already occupied")
        self.board[row, col] = self.current_player
        self.last_row, self.last_col = row, col

        # Rotate the board part
        self.rotate_board_part(rotation // 2, rotation % 2 == 0)  # Adjust rotation values

        # Check for a win, draw, or continue the game
        reward, done, info = self.get_reward_done_info(action) # Pass action to reward function

        # Switch to the next player's turn
        if not done:
            self.current_player = 3 - self.current_player  # Switch between player 1 and player 2

        return self.board, reward, done, info

    def get_reward_done_info(self, action):
        # Calculate immediate reward based on placed piece
        progress_reward = self.calculate_progress_reward(self.current_player, self.last_row, self.last_col)

        block_opponent_win_reward = self.check_block_opponent_win() * 60 # Reward for blocking opponent win (weight = 60)
        create_winning_threat_reward = self.check_create_winning_threat() * 20 # Reward for creating winning threat (weight = 20)
        give_opponent_winning_move_penalty = self.check_give_opponent_winning_move() * -80 # Penalty for giving opponent winning move (weight = -80)


        # Check for win/draw
        win_player = None
        if self.check_win(self.current_player): # Optimized: Check win only for current player first
            win_player = self.current_player
        elif self.check_win(3 - self.current_player): # Then check for opponent
            win_player = 3 - self.current_player
        elif self.check_win(1) and self.check_win(2): # Double check for draw condition if needed (less frequent)
            return 50.0, True, {'winner': 'Draw'}

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
        last_row, last_col = self.get_last_move()
        if last_row is None or last_col is None: # No move made yet, no win possible
            return False

        # Check in all directions from the last placed piece
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self.count_aligned(last_row, last_col, dr, dc, player) >= 5:
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
        for action in range(36 * 8): # Iterate through all possible actions
            if self.is_valid_action(action):
                valid_actions_list.append(action)
        return valid_actions_list

    def is_valid_action(self, action):
        # Decode the action into board_button and rotation using divmod
        board_button, _ = divmod(action, 8)
        row, col = divmod(board_button, 6)
        return self.board[row, col] == 0

    def action_to_move(self, action):
        # Convert the flattened action index to the corresponding (row, col, rotation) values
        board_button, rotation = divmod(action, 8)
        row, col = divmod(board_button, 6)
        return row, col, rotation // 2, rotation % 2 == 0

    def clone(self):
        new_env = PentagoEnv2()
        new_env.board = self.board.clone()
        new_env.current_player = self.current_player
        new_env.winner = self.winner
        new_env.last_row = self.last_row
        new_env.last_col = self.last_col
        new_env.previous_board = self.previous_board.clone() if self.previous_board is not None else None # Clone previous board as well
        return new_env

    def check_block_opponent_win(self):
        opponent_player = 3 - self.current_player
        opponent_winning_moves_before = self._get_winning_moves_optimized(opponent_player, self.previous_board) if self.previous_board is not None else [] # Optimized version
        opponent_winning_moves_after = self._get_winning_moves_optimized(opponent_player, self.board) # Optimized version

        if opponent_winning_moves_before and not opponent_winning_moves_after:
            return 1 # Blocked opponent win
        return 0

    def check_create_winning_threat(self):
        current_player_winning_moves = self._get_winning_moves_optimized(self.current_player, self.board) # Optimized version
        if current_player_winning_moves:
            return 1 # Created winning threat (or immediate win)
        return 0

    def check_give_opponent_winning_move(self):
        opponent_player = 3 - self.current_player
        opponent_winning_moves = self._get_winning_moves_optimized(opponent_player, self.board) # Optimized version
        if opponent_winning_moves:
            return 1 # Gave opponent winning move
        return 0

    def _get_winning_moves_optimized(self, player, current_board): # Optimized _get_winning_moves - No Board Cloning Version
        winning_moves = []
        original_board = self.board.clone() # Clone board *once* outside loops

        for action in range(36 * 8):
            board_button, rotation = divmod(action, 8)
            row, col = divmod(board_button, 6)
            if current_board[row, col] == 0:
                # 1. Temporarily modify the board *IN-PLACE*
                original_subgrid = None
                corner_index = rotation // 2
                if not (corner_index < 0 or corner_index > 3):
                    row_range, col_range = self.get_corner_ranges(corner_index)
                    subgrid_to_rotate = self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]].clone() # Clone subgrid only
                    original_subgrid = subgrid_to_rotate.clone() # Backup subgrid

                self.board = current_board.clone() # Still need to set board from current_board for each action
                self.board[row, col] = player

                self.rotate_board_part(rotation // 2, rotation % 2 == 0) # Rotate IN-PLACE

                # 2. Check for win (no change here - using optimized check_win already)
                if self.check_win(player):
                    winning_moves.append(action)

                # 3. Revert board changes (restore from original_board - NO CLONE here)
                self.board = original_board.clone() # Restore original board for next iteration - still need to clone for next iteration to be clean
                if original_subgrid is not None:
                    row_range, col_range = self.get_corner_ranges(corner_index) # need to get row_range and col_range again
                    self.board[row_range[0]:row_range[1], col_range[0]:col_range[1]] = original_subgrid # Restore subgrid

        return winning_moves

if __name__ == '__main__':
    env = PentagoEnv2()

    num_repetitions = 10

    functions_to_test = [
        "reset",
        "step",
        "get_reward_done_info",
        "calculate_progress_reward",
        "render",
        "check_win",
        "count_aligned",
        "count_direction",
        "rotate_board_part",
        "get_corner_ranges",
        "check_draw",
        "get_last_move",
        "get_valid_actions",
        "is_valid_action",
        "action_to_move",
        "clone",
        "check_block_opponent_win",
        "check_create_winning_threat",
        "check_give_opponent_winning_move",
        "_get_winning_moves_optimized", # Profiling optimized _get_winning_moves
        "_get_winning_moves" # Profiling original _get_winning_moves
    ]

    execution_times = {}

    for func_name in functions_to_test:
        if func_name == "reset":
            time = timeit.timeit(lambda: env.reset(), number=num_repetitions)
        elif func_name == "step":
            time = timeit.timeit(lambda: (env.reset(), env.step(0))[1], number=num_repetitions)
        elif func_name == "get_reward_done_info":
            time = timeit.timeit(lambda: env.get_reward_done_info(0), number=num_repetitions)
        elif func_name == "calculate_progress_reward":
            time = timeit.timeit(lambda: env.calculate_progress_reward(1, 0, 0), number=num_repetitions)
        elif func_name == "render":
            time = timeit.timeit(lambda: env.render(), number=num_repetitions)
            num_repetitions = 1
        elif func_name == "check_win":
            time = timeit.timeit(lambda: env.check_win(1), number=num_repetitions)
        elif func_name == "count_aligned":
            time = timeit.timeit(lambda: env.count_aligned(0, 0, 1, 0, 1), number=num_repetitions)
        elif func_name == "count_direction":
            time = timeit.timeit(lambda: env.count_direction(0, 0, 1, 0, 1, 1), number=num_repetitions)
        elif func_name == "rotate_board_part":
            time = timeit.timeit(lambda: env.rotate_board_part(0, True), number=num_repetitions)
        elif func_name == "get_corner_ranges":
            time = timeit.timeit(lambda: env.get_corner_ranges(0), number=num_repetitions)
        elif func_name == "check_draw":
            time = timeit.timeit(lambda: env.check_draw(), number=num_repetitions)
        elif func_name == "get_last_move":
            time = timeit.timeit(lambda: env.get_last_move(), number=num_repetitions)
        elif func_name == "get_valid_actions":
            time = timeit.timeit(lambda: env.get_valid_actions(), number=num_repetitions)
        elif func_name == "is_valid_action":
            time = timeit.timeit(lambda: env.is_valid_action(0), number=num_repetitions)
        elif func_name == "action_to_move":
            time = timeit.timeit(lambda: env.action_to_move(0), number=num_repetitions)
        elif func_name == "clone":
            time = timeit.timeit(lambda: env.clone(), number=num_repetitions)
        elif func_name == "check_block_opponent_win":
            env.reset()
            env.step(0)
            time = timeit.timeit(lambda: env.check_block_opponent_win(), number=num_repetitions)
        elif func_name == "check_create_winning_threat":
            time = timeit.timeit(lambda: env.check_create_winning_threat(), number=num_repetitions)
        elif func_name == "check_give_opponent_winning_move":
            time = timeit.timeit(lambda: env.check_give_opponent_winning_move(), number=num_repetitions)
        elif func_name == "_get_winning_moves_optimized":
            time = timeit.timeit(lambda: env._get_winning_moves_optimized(1, env.board), number=num_repetitions)
        elif func_name == "_get_winning_moves":
            time = timeit.timeit(lambda: env._get_winning_moves(1, env.board), number=num_repetitions)
        else:
            time = -1

        execution_times[func_name] = time

    print("Execution times for PentagoEnv2 (lower is better, in seconds for {} repetitions):".format(num_repetitions))
    for func_name, time in execution_times.items():
        print(f"- {func_name}: {time:.6f} seconds")