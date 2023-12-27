import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
from environment import PentagoEnv

# Define the Dueling DQN model using PyTorch
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        # Shared layers
        self.fc1_shared = nn.Linear(6 * 6 * 3, 256)
        self.fc2_shared = nn.Linear(256, 128)
        
        # Value stream layers
        self.fc3_value = nn.Linear(128, 64)
        self.fc4_value = nn.Linear(64, 1)

        # Advantage stream layers for board button actions
        self.fc3_advantage_board = nn.Linear(128, 64)
        self.fc4_advantage_board = nn.Linear(64, 36)  # Assuming there are 36 board button actions

        # Advantage stream layers for rotation actions
        self.fc3_advantage_rotation = nn.Linear(128, 64)
        self.fc4_advantage_rotation = nn.Linear(64, 8)  # Assuming there are 8 rotation actions

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 6 * 6 * 3)

        # Shared layers
        x_shared = F.relu(self.fc1_shared(x))
        x_shared = F.relu(self.fc2_shared(x_shared))

        # Value stream
        x_value = F.relu(self.fc3_value(x_shared))
        value = self.fc4_value(x_value)

        # Advantage stream for board button actions
        x_advantage_board = F.relu(self.fc3_advantage_board(x_shared))
        advantage_board = self.fc4_advantage_board(x_advantage_board)

        # Advantage stream for rotation actions
        x_advantage_rotation = F.relu(self.fc3_advantage_rotation(x_shared))
        advantage_rotation = self.fc4_advantage_rotation(x_advantage_rotation)

        # Combine value and advantage to get Q-values for board button actions
        q_values_board = value + (advantage_board - advantage_board.mean(dim=1, keepdim=True))

        # Combine value and advantage to get Q-values for rotation actions
        q_values_rotation = value + (advantage_rotation - advantage_rotation.mean(dim=1, keepdim=True))

        return q_values_board, q_values_rotation

# Implement experience replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        # Add an experience to the buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Sample a batch of experiences from the buffer
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        # Return the current size of the internal buffer
        return len(self.buffer)

# Define the DQN agent
class DDQNAgent:
    def __init__(self, env, buffer_capacity=1000000, batch_size=64, target_update_frequency=10):
        self.env = env
        self.model = DuelingDQN()  # Change here
        self.target_model = DuelingDQN()  # Change here
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.num_training_steps = 0

    def select_action(self, state, epsilon):
        # Directly check which columns are not full
        available_actions = self.env.get_valid_actions()

        # Ensure the model is in evaluation mode
        self.model.eval()

        if random.random() < epsilon:
            board_action = random.choice(available_actions)
            rotation_action = random.randint(0, 7)  # Assuming rotation actions are integers from 0 to 7
        else:
            state_tensor = state.unsqueeze(0)
            with torch.no_grad():
                q_values_board, q_values_rotation = self.model(state_tensor)
                q_values_board = q_values_board.squeeze()

            # Mask the Q-values of invalid actions with a very negative number
            masked_board_q_values = torch.full(q_values_board.shape, float('-inf'))
            masked_board_q_values[available_actions] = q_values_board[available_actions]

            # Get the board action with the highest Q-value among the valid actions
            board_action = torch.argmax(masked_board_q_values).item()

            # Get the rotation action with the highest Q-value
            rotation_action = torch.argmax(q_values_rotation).item()

        # Ensure the model is back in training mode
        self.model.train()

        return board_action, rotation_action

    def train_step(self):
        if len(self.buffer) >= self.batch_size:
            
            experiences = list(self.buffer.sample(self.batch_size))  # Convert to list for better indexing
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = torch.stack(states)
            next_states = torch.stack(next_states)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Use target model for action selection in Double Q-learning
            target_actions_board = self.model(next_states)[0].max(1)[1].unsqueeze(-1)
            target_actions_rotation = self.model(next_states)[1].max(1)[1].unsqueeze(-1)
            
            max_next_q_values_board = self.target_model(next_states)[0].gather(1, target_actions_board).squeeze(-1)
            max_next_q_values_rotation = self.target_model(next_states)[1].gather(1, target_actions_rotation).squeeze(-1)

            current_q_values_board = self.model(states)[0].gather(1, actions[:, 0].unsqueeze(-1)).squeeze(-1)
            current_q_values_rotation = self.model(states)[1].gather(1, actions[:, 1].unsqueeze(-1)).squeeze(-1)
            
            expected_q_values_board = rewards + (1 - dones) * 0.99 * max_next_q_values_board  # Assuming a gamma of 0.99
            expected_q_values_rotation = rewards + (1 - dones) * 0.99 * max_next_q_values_rotation  # Assuming a gamma of 0.99

            loss_board = self.loss_fn(current_q_values_board, expected_q_values_board)
            loss_rotation = self.loss_fn(current_q_values_rotation, expected_q_values_rotation)

            self.optimizer.zero_grad()
            (loss_board + loss_rotation).backward()
            self.optimizer.step()

            if self.num_training_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.num_training_steps += 1

# Training loop
def agent_vs_agent_train(agents, env, num_episodes=1000, epsilon_start=0.5, epsilon_final=0.01, epsilon_decay=0.999):
    epsilon = epsilon_start
    
    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training", unit="episode"):
        state = env.reset()
        total_rewards = [0, 0]
        done = False

        while not done:
            for i in range(len(agents)):
                action = agents[i].select_action(state, epsilon)
                next_state, reward, done, info = env.step(action)
                total_rewards[i] += reward
                agents[i].buffer.add(Experience(state, action, reward, next_state, done))
                state = next_state
                if done:
                    break

        # Batch processing of experiences for each agent
        for agent in agents:
            agent.train_step()

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Player 1: Reward {total_rewards[0]}, Player 2: Reward {total_rewards[1]}, Epsilon: {epsilon:.2f}")

        # Decay epsilon for the next episode
        epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

# Function to load a saved agent for DDQNAgent
def load_ddqn_agent(agent, checkpoint_path, player_name):
    checkpoint = torch.load(checkpoint_path)
    agent.model.load_state_dict(checkpoint[f'model_state_dict_{player_name}'])
    agent.target_model.load_state_dict(checkpoint[f'target_model_state_dict_{player_name}'])
    agent.optimizer.load_state_dict(checkpoint[f'optimizer_state_dict_{player_name}'])

# Example usage:
if __name__ == '__main__':
    env = PentagoEnv()  # Assuming PentagoGame implements the necessary environment methods

    # Players
    ddqn_agents = [DDQNAgent(env), DDQNAgent(env)]

    # Load pre-trained agents
    checkpoint_path = 'saved_agents/ddqnd_agents_after_train.pth'
    for i, agent in enumerate(ddqn_agents):
        load_ddqn_agent(agent, checkpoint_path, f'player{i + 1}')

    # Continue training
    agent_vs_agent_train(ddqn_agents, env, num_episodes=100000)

    # Save the trained agents
    torch.save({
        'model_state_dict_player1': ddqn_agents[0].model.state_dict(),
        'target_model_state_dict_player1': ddqn_agents[0].target_model.state_dict(),
        'optimizer_state_dict_player1': ddqn_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': ddqn_agents[1].model.state_dict(),
        'target_model_state_dict_player2': ddqn_agents[1].target_model.state_dict(),
        'optimizer_state_dict_player2': ddqn_agents[1].optimizer.state_dict(),
    }, 'saved_agents/ddqnd_agents_after_continue_train.pth')

