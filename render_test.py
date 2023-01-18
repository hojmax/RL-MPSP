from env import MPSPEnv
import numpy as np
import gym
import torch

env = MPSPEnv(10, 4, 10)

env = gym.wrappers.RecordVideo(env, video_folder=f"video/test")
obs = env.reset()

done = False
while not done:
    action_mask = env.action_masks()

    # Take random action
    action_p = action_mask / np.sum(action_mask)
    action = np.random.choice(np.arange(len(action_mask)), p=action_p)

    env.unwrapped.probs = torch.from_numpy(action_p)
    env.unwrapped.prev_action = action
    env.unwrapped.action_mask = action_mask

    env.render()
    obs, reward, done, _ = env.step(action)
