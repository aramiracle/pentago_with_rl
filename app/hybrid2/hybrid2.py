import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
import numpy as np
from app.environment2 import PentagoEnv2

# Define the CNN-based Dueling DQN model using PyTorch
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        # Convolutional layers for shared feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # Input channels: 3 (planes), Output channels: 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the output size after convolutions and flatten
        # Assuming input is 6x6, after padding=1 and kernel_size=3, size remains roughly 6x6
        conv_out_size = 6 * 6 * 64  # Height * Width * Channels after conv layers

        # Shared layers after convolution
        self.fc_shared = nn.Linear(conv_out_size, 512)
        self.bn_shared = nn.BatchNorm1d(512)

        # Combined stream layers - for single output
        self.fc_combined1 = nn.Linear(512, 256)
        self.bn_combined1 = nn.BatchNorm1d(256)
        self.fc_combined2 = nn.Linear(256, 36 * 8) # 36 board actions * 8 rotations

    def forward(self, x):
        # Input x should be (batch_size, 6, 6) representing the board state
        # Convert to one-hot encoding and reshape to (batch_size, 3, 6, 6)
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float() # (batch_size, 6, 6, 3)
        x = x.permute(0, 3, 1, 2) # Reshape to (batch_size, 3, 6, 6) - channels first

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the convolutional output
        x_shared = x.reshape(x.size(0), -1) # (batch_size, conv_out_size)

        # Shared fully connected layer
        x_shared = F.relu(self.bn_shared(self.fc_shared(x_shared)))

        # Combined stream - single output
        x_combined = F.relu(self.bn_combined1(self.fc_combined1(x_shared)))
        q_values_combined = self.fc_combined2(x_combined) # Q-values for all combined actions

        return q_values_combined

# Implement Prioritized experience replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedExperienceReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        self.capacity = capacity

    def add(self, experience, priority):
        max_prio = max(self.priorities) if self.buffer else 1.0 # Initial max priority if buffer is empty is 1

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = self.beta

        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1) # Shape to [batch_size, 1]

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = abs(err) + 1e-6  # Small positive constant to avoid zero priority

    def update_beta(self, frame_idx):
        beta_increment_per_sampling = (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(1.0, self.beta_start + beta_increment_per_sampling * frame_idx)

    def __len__(self):
        return len(self.buffer)

# Define the DQN agent
class Hybrid2Agent:
    def __init__(self, env, buffer_capacity=1000000, batch_size=64, target_update_frequency=10):
        self.env = env
        self.model = DuelingDQN()  # Now using CNN DuelingDQN
        self.target_model = DuelingDQN()  # Now using CNN DuelingDQN
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = PrioritizedExperienceReplayBuffer(buffer_capacity) # Use PrioritizedExperienceReplayBuffer
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4) # Adjusted learning rate
        self.loss_fn = nn.MSELoss()
        self.num_training_steps = 0
        self.frame_idx = 0 # Frame index for beta update in PER

    def select_action(self, state, epsilon):
        # Directly check which columns are not full
        available_actions = self.env.get_valid_actions()

        # Ensure the model is in evaluation mode
        self.model.eval()

        if random.random() < epsilon:
            action = random.choice(available_actions)
        else:
            # Check for instant win moves
            instant_win_actions = []
            for action in available_actions:
                if self.is_instant_win(self.env, action):
                    instant_win_actions.append(action)
                    break
            if instant_win_actions:
                # If there are instant win moves, choose one randomly
                action = random.choice(instant_win_actions)
            else:
                state_tensor = state.unsqueeze(0)  # Adding batch dimension
                with torch.no_grad():
                    q_values_combined = self.model(state_tensor).squeeze()

                # Mask the Q-values of invalid actions with a very negative number
                masked_q_values = torch.full(q_values_combined.shape, float('-inf'))
                masked_q_values[available_actions] = q_values_combined[available_actions]

                # Get the action with the highest Q-value among the valid actions
                action = torch.argmax(masked_q_values).item()

        # Ensure the model is back in training mode
        self.model.train()

        return action

    def is_instant_win(self, env, action):
        # Check if the agent has an instant winning move in the next turn
        next_env = env.clone()
        _, reward, _, _ = next_env.step(action)  # Only need reward
        return reward == 1

    def add_experience(self, state, action, reward, next_state, done):
        priority = max(self.buffer.priorities) if self.buffer else 1.0
        self.buffer.add(Experience(state, action, reward, next_state, done), priority)

    def train_step(self):
        if len(self.buffer) >= self.batch_size:
            self.frame_idx += 1
            self.buffer.update_beta(self.frame_idx)

            experiences, indices, weights = self.buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = torch.stack(states) # states are (6, 6) tensors now
            next_states = torch.stack(next_states) # next_states are (6, 6) tensors now
            rewards = torch.tensor(rewards, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1) # actions are now single combined actions
            dones = torch.tensor(dones, dtype=torch.float32)
            weights = torch.tensor(weights).squeeze() # Ensure weights is a tensor and squeeze to correct shape

            # Use target model for action selection in Double Q-learning
            target_actions = self.model(next_states).max(1)[1].unsqueeze(-1)
            max_next_q_values = self.target_model(next_states).gather(1, target_actions).squeeze(-1)

            current_q_values = self.model(states).gather(1, actions)
            expected_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values

            # Calculate TD errors for priority update
            td_errors = expected_q_values - current_q_values.squeeze(-1)

            # Weighted MSE Loss with PER importance sampling weights
            loss = (weights * self.loss_fn(current_q_values, expected_q_values)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10) # Gradient clipping
            self.optimizer.step()

            # Update priorities in buffer
            self.buffer.update_priorities(indices, td_errors.cpu().detach().numpy())

            if self.num_training_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.num_training_steps += 1

    def add_experience(self, state, action, reward, next_state, done):
        priority = max(self.buffer.priorities) if self.buffer else 1.0
        self.buffer.add(Experience(state, action, reward, next_state, done), priority)

# Training loop
def agent_vs_agent_train(agents, env, num_episodes=1000, epsilon_start=0.5, epsilon_final=0.01, epsilon_decay=0.999):
    epsilon = epsilon_start
    episode_rewards_player1 = deque(maxlen=100) # Track rewards for player 1
    episode_rewards_player2 = deque(maxlen=100) # Track rewards for player 2

    for episode in tqdm(range(num_episodes), desc="Hybrid2 Agent vs Agent Training (CNN)", unit="episode"): # Changed desc
        state = env.reset()
        state = torch.tensor(state, dtype=torch.int) # Ensure state is a tensor
        total_rewards = [0, 0]
        done = False

        while not done:
            for i in range(len(agents)):
                action = agents[i].select_action(state, epsilon)
                next_state, reward, done, info = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.int) # Ensure next_state is a tensor
                total_rewards[i] += reward
                agents[i].add_experience(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

        # Batch processing of experiences for each agent
        for agent in agents:
            agent.train_step()

        episode_rewards_player1.append(total_rewards[0]) # Append reward for player 1
        episode_rewards_player2.append(total_rewards[1]) # Append reward for player 2

        avg_reward_player1 = np.mean(episode_rewards_player1) if episode_rewards_player1 else 0
        avg_reward_player2 = np.mean(episode_rewards_player2) if episode_rewards_player2 else 0

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Player 1: Reward {total_rewards[0]}, Player 2: Reward {total_rewards[1]}, Epsilon: {epsilon:.2f}, Beta: {agents[0].buffer.beta:.2f}")

        if episode % 100 == 0 and episode > 0:
            tqdm.write(f"--- Episode {episode} --- Avg Reward (Last 100 Episodes) - Player 1: {avg_reward_player1:.2f}, Player 2: {avg_reward_player2:.2f} ---")

        # Decay epsilon for the next episode
        epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

# Function to load a saved agent
def load_agent(agent, checkpoint_path, player_name):
    checkpoint = torch.load(checkpoint_path)
    agent.model.load_state_dict(checkpoint[f'model_state_dict_{player_name}'])
    agent.target_model.load_state_dict(checkpoint[f'target_model_state_dict_{player_name}'])
    agent.optimizer.load_state_dict(checkpoint[f'optimizer_state_dict_{player_name}'])

# Example usage:
if __name__ == '__main__':
    env = PentagoEnv2()  # Assuming PentagoEnv2 implements the necessary environment methods

    # Hyperparameters - Tune these!
    learning_rate = 1e-4
    epsilon_start = 0.5
    epsilon_final = 0.01
    epsilon_decay = 0.995
    num_episodes = 1000
    batch_size = 128
    target_update_frequency = 500
    buffer_capacity = 1000000

    # Players
    dqn_agents = [Hybrid2Agent(env), Hybrid2Agent(env)]

    # Load pre-trained agents - you might want to remove this for training from scratch
    # checkpoint_path = 'saved_agents/hybrid2_agents_per_bn_deeper_trained.pth'
    # for i, agent in enumerate(dqn_agents):
    #     load_agent(agent, checkpoint_path, f'player{i + 1}')

    print('Agents Loaded.')

    # Continue training
    agent_vs_agent_train(dqn_agents, env, num_episodes=30000, epsilon_start=epsilon_start, epsilon_final=epsilon_final, epsilon_decay=epsilon_decay)

    # Save the trained agents
    os.makedirs('saved_agents', exist_ok=True)
    torch.save({
        'model_state_dict_player1': dqn_agents[0].model.state_dict(),
        'target_model_state_dict_player1': dqn_agents[0].target_model.state_dict(),
        'optimizer_state_dict_player1': dqn_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': dqn_agents[1].model.state_dict(),
        'target_model_state_dict_player2': dqn_agents[1].model.state_dict(),
        'optimizer_state_dict_player2': dqn_agents[1].optimizer.state_dict(),
    }, 'saved_agents/hybrid2_agents_trained.pth') # Changed save path
    print(f"Trained CNN-based Hybrid2 agents with PER, BN, Deeper Network saved to 'saved_agents/hybrid2_agents_trained.pth'")