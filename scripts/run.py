import os
import time
import numpy as np
import torch
import torch.nn as nn
import pygame

from env.gym_env import MyPacman
from env.utils import get_valid_mask

device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

# Get action from model
def get_action(env, model, state):
    action_mask = get_valid_mask(env)
    valid_actions = np.where(action_mask == 1)[0]
    if len(valid_actions) == 0:
        return 0
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(state_t).squeeze(0).cpu().numpy()
        q_values[action_mask == 0] = -1e9
        return int(np.argmax(q_values))

# Main
def main():
    pygame.init()
    env = MyPacman()
    state = env.reset()

    obs_size = len(state)
    n_actions = env.num_actions
    hidden_layer = 256
    model = DuelingNetwork(obs_size, n_actions, hidden_layer).to(device)

    #load model
    model_path = os.path.join(os.path.dirname(__file__), "models", "rl_pacman.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    done = False
    total_reward = 0.0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        action = get_action(env, model, state)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        total_reward += reward

        env.render(scale=28)
        time.sleep(0.08)

    print("\nGame finished.")
    print(f"Reward  : {total_reward:.2f}")
    print(f"Score   : {env.score:.2f}")

    pygame.quit()

if __name__ == "__main__":
    main()