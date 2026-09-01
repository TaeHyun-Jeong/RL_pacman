import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from env.gym_env import MyPacman
from env.utils import get_valid_mask, obs_to_action_mask

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Hyperparameters
num_episodes =8000
GAMMA = 0.99
learning_rate = 0.0005

hidden_layer = 256
replay_memory_size = 100000
batch_size = 128
target_nn_update_frequency = 2000
train_start_size = 2000
e_start = 1.0
e_end = 0.05
e_decay = 300000

env = MyPacman()
state = env.reset()
obs_size = len(state)
n_actions = env.num_actions

# Replay Memory
class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory): self.memory.append(transition)
        else: self.memory[self.position] = transition
            
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(next_states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
    )

    def __len__(self):
        return len(self.memory)
    
# Dueling Network
class DuelingNetwork(nn.Module):
    def __init__(self, input_dim, n_outputs, hidden_layer):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_layer),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer // 2),
            nn.ReLU(),
            nn.Linear(hidden_layer // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer // 2),
            nn.ReLU(),
            nn.Linear(hidden_layer // 2, n_outputs),
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# Setup
memory = ExperienceReplay(replay_memory_size)
Q = DuelingNetwork(obs_size, n_actions, hidden_layer).to(device)
target_Q = DuelingNetwork(obs_size, n_actions, hidden_layer).to(device)
target_Q.load_state_dict(Q.state_dict())

optimizer = optim.Adam(Q.parameters(), lr=learning_rate)
total_steps = 0
reward_sum = 0.0

# Select action
def select_action(state, steps_done):
    e_threshold = e_end + (e_start - e_end) * math.exp(-1.0 * steps_done / e_decay)

    action_mask = get_valid_mask(env)
    valid_actions = np.where(action_mask == 1)[0]

    if len(valid_actions) == 0:
        return random.randrange(n_actions), e_threshold
    
    if random.random() < e_threshold:
        return int(np.random.choice(valid_actions)), e_threshold

    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = Q(state).squeeze(0).detach().cpu().numpy()
        q_values[action_mask == 0] = -1e9
        action = int(np.argmax(q_values))
        return action, e_threshold

# Train loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        total_steps += 1

        action, epsilon = select_action(state, total_steps)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, next_state, reward, done)

        state = next_state
        total_reward += reward

        if len(memory) >= max(batch_size, train_start_size):
            states, actions, next_states, rewards, dones = memory.sample(batch_size)

            states = torch.tensor(states, dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
            dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

            y_pred = Q(states).gather(1, actions)

            with torch.no_grad():
                online_next_q = Q(next_states).detach().cpu().numpy()

                next_masks = []
                for next_obs in next_states.detach().cpu().numpy():
                    next_mask = obs_to_action_mask(env, next_obs)
                    next_masks.append(next_mask)
                next_masks = np.array(next_masks)

                online_next_q[next_masks == 0] = -1e9
                next_actions = np.argmax(online_next_q, axis=1)
                next_actions = torch.tensor(next_actions, dtype=torch.int64, device=device).unsqueeze(1)

                target_next_q_values = target_Q(next_states).gather(1, next_actions)

                no_valid_next = (next_masks.sum(axis=1) == 0)
                no_valid_next = torch.tensor(no_valid_next, dtype=torch.bool, device=device).unsqueeze(1)
                target_next_q_values = torch.where(
                    no_valid_next,
                    torch.zeros_like(target_next_q_values),
                    target_next_q_values
                )
                y_target = rewards + (1 - dones) * GAMMA * target_next_q_values

            loss = ((y_pred - y_target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=10.0)
            optimizer.step()

        if total_steps % target_nn_update_frequency == 0:
            target_Q.load_state_dict(Q.state_dict())

    reward_sum += total_reward
    if (episode + 1) % 10 == 0:
        avg_reward = reward_sum / 10
        print(
            f"Episode {episode + 1}, "
            f"AvgReward(10): {avg_reward:.2f}, "
            f"Epsilon: {epsilon:.4f}"
        )
        reward_sum = 0.0

torch.save(Q.state_dict(), os.path.join(os.path.dirname(__file__), "..", "models", "rl_pacman.pth"))
print("Model saved.")