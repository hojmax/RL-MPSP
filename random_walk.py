from env import MPSPEnv
from benchmark import get_benchmarking_data
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import gym
import wandb
# wandb.init(monitor_gym=True)

env = MPSPEnv(10, 4, 10)


rewards = []

for i in range(100):

    env.reset()
    print(env.transportation_matrix)
    cum_reward = 0
    done = False
    while not done:
        
        # take a random action
        action_mask = env.action_masks()
        action_p = action_mask / np.sum(action_mask)
        
        action = np.random.choice(np.arange(len(action_mask)), p=action_p)
        obs, reward, done, _ = env.step(action)
        cum_reward += reward

    rewards.append(cum_reward)


print(f'Average reward: {np.mean(rewards)}')
print(f'Max reward: {np.max(rewards)}')
print(f'Min reward: {np.min(rewards)}')
print(f'Rewards above -10: {np.sum(np.array(rewards) > -10)}')


    