from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
from sb3_contrib.ppo_mask import MaskablePPO
from benchmark import get_benchmarking_data
from env import MPSPEnv
import numpy as np
import torch
import wandb
import os
import gym
from tqdm import tqdm

wandb.login()

config = {
    # Environment
    'ROWS': 10,
    'COLUMNS': 4,
    'N_PORTS': 10,
    # Model
    'PI_LAYER_SIZES': [64, 128, 64],
    'VF_LAYER_SIZES': [64, 128, 64],
    # Training
    'TOTAL_TIMESTEPS': 4800000,
    'BATCH_SIZE': 128
}

run = wandb.init(
    project="PPO-SB3",
    entity="rl-msps",
    sync_tensorboard=True,
    name=f"N{config['N_PORTS']}_R{config['ROWS']}_C{config['COLUMNS']}",
    config=config,
    tags=["test"],
    monitor_gym=True,
)

env = make_vec_env(
    lambda: MPSPEnv(
        config['ROWS'],
        config['COLUMNS'],
        config['N_PORTS']
    ),
    n_envs=8  # M2 with 8 cores
)


policy_kwargs = {
    'activation_fn': torch.nn.ReLU,
    'net_arch': [{
        'pi': config['PI_LAYER_SIZES'],
        'vf': config['VF_LAYER_SIZES']
    }]
}

wandb_run_path = None

if wandb_run_path:
    model_file = wandb.restore('model.zip', run_path=wandb_run_path)
    model = MaskablePPO.load(
        model_file.name,
        env=env
    )
else:
    model = MaskablePPO(
        policy='MultiInputPolicy',
        env=env,
        batch_size=config['BATCH_SIZE'],
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs=policy_kwargs
    )


    model.learn(
        total_timesteps=config['TOTAL_TIMESTEPS'],
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            model_save_freq=config['TOTAL_TIMESTEPS'] // 4,
        ),
        progress_bar=True
    )

eval_data = get_benchmarking_data('rl-mpsp-benchmark/set_2')
eval_data = [
    e for e in eval_data if (
        e['R'] == config['ROWS'] and
        e['C'] == config['COLUMNS'] and
        e['N'] == config['N_PORTS']
    )
]

# Creating seperate env for evaluation
env = MPSPEnv(
    config['ROWS'],
    config['COLUMNS'],
    config['N_PORTS']
)
env = gym.wrappers.RecordVideo(env, video_folder='video', step_trigger=lambda x: True)

eval_rewards = []
# Negative because env returns negative reward for shifts
paper_rewards = [-e['paper_result'] for e in eval_data]
paper_seeds = [e['seed'] for e in eval_data]

for e in tqdm(eval_data, desc='Evaluating'):
    total_reward = 0
    obs = env.reset(
        transportation_matrix=e['transportation_matrix']
    )
    done = False
    while not done:
        action, _ = model.predict(
            obs,
            action_masks=env.action_masks()
        )
        env.render()
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    eval_rewards.append(total_reward)

eval = {
    'mean_reward': np.mean(eval_rewards),
    'mean_paper_reward': np.mean(paper_rewards),
    'rewards': eval_rewards,
    'paper_rewards': paper_rewards,
    'paper_seeds': paper_seeds
}
run.summary['evaluation_benchmark'] = eval

run.finish()