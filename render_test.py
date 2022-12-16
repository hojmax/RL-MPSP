from env import MPSPEnv
from benchmark import get_benchmarking_data
import numpy as np
import pygame
import gym
import wandb
wandb.init(monitor_gym=True)

env = MPSPEnv(10, 10, 10)
# env = gym.wrappers.Monitor(env, './video', video_callable=lambda episode_id: True, force=True)
env = gym.wrappers.RecordVideo(env, video_folder='video', step_trigger=lambda x: True)


config = {
    # Environment
    'ROWS': 10,
    'COLUMNS': 4,
    'N_PORTS': 10,
    # Model
    'PI_LAYER_SIZES': [64, 128, 64],
    'VF_LAYER_SIZES': [64, 128, 64],
    # Training
    'TOTAL_TIMESTEPS': 48000,
    'BATCH_SIZE': 128
}


eval_data = get_benchmarking_data('rl-mpsp-benchmark/set_2')
eval_data = [
    e for e in eval_data if (
        e['R'] == config['ROWS'] and
        e['C'] == config['COLUMNS'] and
        e['N'] == config['N_PORTS']
    )
]

# Pick a random sample from evaluation data
sample = eval_data[np.random.randint(len(eval_data))]
obs = env.reset(
    transportation_matrix=sample['transportation_matrix']
)

# while True:
#     env.render()


done = False
while not done:
    
    # take a random action
    action_mask = env.action_masks()
    action_p = action_mask / np.sum(action_mask)
    
    action = np.random.choice(np.arange(len(action_mask)), p=action_p)
    env.render(probs=action_p, action=action)
    obs, reward, done, _ = env.step(action)
    
# wandb.log({"videos": [wandb.Video("./video/rl-video-step-0.mp4")]})