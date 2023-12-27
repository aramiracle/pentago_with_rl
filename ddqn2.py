import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
from environment2 import PentagoEnv2

# Modify the model output to have a single output for the combined action space
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(6 * 6 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 36 * 8)  # Assuming 36 * 8 possible actions

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.view(-1, 6 * 6 * 3)

        # Shared layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Combined stream
        x = F.relu(self.fc3(x))

        return x

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
class DDQN2Agent:
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
            # Randomly choose an action
            action = random.choice(available_actions)
        else:
            state_tensor = state.unsqueeze(0)
            with torch.no_grad():
                q_values_combined = self.model(state_tensor).squeeze()

            # Mask the Q-values of invalid actions with a very negative number
            masked_q_values = torch.full(q_values_combined.shape, float('-inf'))
            masked_q_values[available_actions] = q_values_combined[available_actions]

            # Get the combined action with the highest Q-value among the valid actions
            action = torch.argmax(masked_q_values).item()

        # Ensure the model is back in training mode
        self.model.train()

        return action

    def train_step(self):
        if len(self.buffer) >= self.batch_size:
            experiences = list(self.buffer.sample(self.batch_size))
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = torch.stack(states)
            next_states = torch.stack(next_states)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            
            # Convert actions to tensor and reshape to (batch_size, 1)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)

            dones = torch.tensor(dones, dtype=torch.float32)

            # Use target model for action selection in Double Q-learning
            target_actions = self.model(next_states).max(1)[1].unsqueeze(-1)
            max_next_q_values = self.target_model(next_states).gather(1, target_actions).squeeze(-1)

            current_q_values = self.model(states).gather(1, actions)
            expected_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values

            loss = self.loss_fn(current_q_values, expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
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

# Example usage:
if __name__ == '__main__':
    env = PentagoEnv2() # Assuming PentagoGame implements the necessary environment methods

    # Players
    dqn_agents = [DDQN2Agent(env), DDQN2Agent(env)]

    # Agent vs Agent Training
    agent_vs_agent_train(dqn_agents, env, num_episodes=100000)

    os.makedirs('saved_agents', exist_ok=True)

    # Save the trained agents
    torch.save({
        'model_state_dict_player1': dqn_agents[0].model.state_dict(),
        'target_model_state_dict_player1': dqn_agents[0].target_model.state_dict(),
        'optimizer_state_dict_player1': dqn_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': dqn_agents[1].model.state_dict(),
        'target_model_state_dict_player2': dqn_agents[1].target_model.state_dict(),
        'optimizer_state_dict_player2': dqn_agents[1].optimizer.state_dict(),
    }, 'saved_agents/ddqn2_agents_after_train.pth')
