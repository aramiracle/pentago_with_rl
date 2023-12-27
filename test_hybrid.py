import random
import torch
from tqdm import tqdm
from hybrid import HybridAgent
from environment2 import PentagoEnv2

class RandomBot:
    def __init__(self, env):
        self.env = env

    def select_action(self, state, epsilon):
        available_actions = self.env.get_valid_actions()
        return random.choice(available_actions)
    
def simulate_game(env, player1, player2):
    """Simulates a single game between two AI agents or an AI agent and a random bot."""
    state = env.reset()
    done = False
    while not done:
        if env.current_player == 1:
            action = player1.select_action(state, epsilon=0)
        else:
            action = player2.select_action(state, epsilon=0)
        state, _, done, _ = env.step(action)
    return env.winner

def test_ai_vs_random(env, ai_agent, random_bot, num_games=1000):
    """Tests an AI agent against a random bot over a specified number of games."""
    ai_wins = 0
    random_bot_wins = 0
    draws = 0

    for _ in tqdm(range(num_games), desc='AI vs Random Bot'):
        winner = simulate_game(env, ai_agent, random_bot)
        if winner == 1:
            ai_wins += 1
        elif winner == 2:
            random_bot_wins += 1
        elif winner is None:
            draws += 1

    return ai_wins, random_bot_wins, draws

def test_random_vs_ai(env, random_bot, ai_agent, num_games=1000):
    """Tests a random bot against an AI agent over a specified number of games."""
    ai_wins = 0
    random_bot_wins = 0
    draws = 0

    for _ in tqdm(range(num_games), desc='Random Bot vs AI'):
        winner = simulate_game(env, random_bot, ai_agent)
        if winner == 1:
            random_bot_wins += 1
        elif winner == 2:
             ai_wins += 1
        elif winner is None:
            draws += 1

    return random_bot_wins, ai_wins, draws

if __name__ == '__main__':
    env = PentagoEnv2()

    # Load AI agent
    ai_agent_player1 = HybridAgent(env)
    checkpoint_player1 = torch.load('saved_agents/ddqn2_agents_after_train.pth')
    ai_agent_player1.target_model.load_state_dict(checkpoint_player1['model_state_dict_player1'])

    ai_agent_player2 = HybridAgent(env)
    checkpoint_player2 = torch.load('saved_agents/ddqn2_agents_after_train.pth')
    ai_agent_player2.target_model.load_state_dict(checkpoint_player2['model_state_dict_player2'])

    # Create RandomBot
    random_bot = RandomBot(env)

    # Test scenarios
    ai_vs_random_results = test_ai_vs_random(env, ai_agent_player1, random_bot, num_games=100)
    random_vs_ai_results = test_random_vs_ai(env, random_bot, ai_agent_player2, num_games=100)

    # Print results
    print(f"AI vs Random Bot Results: AI Wins - {ai_vs_random_results[0]}, Random Bot Wins - {ai_vs_random_results[1]}, Draws - {ai_vs_random_results[2]}")
    print(f"Random Bot vs AI Results: Random Bot Wins - {random_vs_ai_results[0]}, AI Wins - {random_vs_ai_results[1]}, Draws - {ai_vs_random_results[2]}")