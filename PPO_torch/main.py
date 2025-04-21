# Abandoned and plan to use stablebaseline.
# This file is a PPO implementation for ContraForce Agent .
# 
# 
import retro
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import cv2
from collections import deque
import PPOAgent
import RolloutBuffer


learning_rate = 2.5e-4
gamma = 0.99
clip_param = 0.2
update_interval = 128
batch_size = 32
ppo_epochs = 4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84))
    obs = obs.astype(np.float32) / 255.0
    return obs 


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])

    return returns


def ppo_update(policy, optimizer, buffer, update_step):
    returns = compute_gae(buffer.next_value, buffer.rewards, buffer.dones, buffer.values)

    states = torch.stack(buffer.states).to(device)    # shape ?
    actions = torch.tensor(buffer.actions).to(device)
    old_logprobs = torch.tensor(buffer.logprobs).to(device)
    returns = torch.tensor(returns).to(device)
    advantages = returns - torch.tensor(buffer.values).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for epoch in range(ppo_epochs):
        idx = np.random.permutation(len(buffer.states))
        #?
    return 



def init_stack(frame):
    stack = deque([frame for _ in range(stack_size)], maxlen=stack_size)
    return stack


def stack_frames(stacked_frames, new_frame):
    stacked_frames.append(new_frame)
    np_stack = np.array(stacked_frames, dtype=np.float32)
    return stacked_frames, np_stack


def train():
    env = retro.make(game='ContraForce-Nes')
    action_dim = env.action_space.shape[0]

    policy = PPOAgent(action_dim, in_channels=stack_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    buffer = RolloutBuffer()

    obs = env.reset()
    # (84,84)
    frame = preprocess(obs)  
    stacked_frames = init_stack(frame)
    #(4,84,84)
    state_np = np.array(stacked_frames, dtype=np.float32)
    # (1,4,84,84)

    return 

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    train()
