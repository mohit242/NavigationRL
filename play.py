import torch
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from navigationrl.agents import DQNAgent


if __name__ == "__main__":
    # Instantiate environment
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    print(env.brain_names)
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qnet_params = {'type': 'simple', 'fc1_units': 128, 'fc2_units': 64}

    agent = DQNAgent(state_size, action_size, 1, device, qnet_params)
    agent.qnet_local.load_state_dict(torch.load("checkpoint.pth"))

    env_info = env.reset(train_mode=False)[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    max_t = 2000
    for t in range(max_t):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        state = next_state
        score += reward
        if done:
            break
    print("Total score: {}".format(score))
