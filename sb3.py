from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
from sb3_contrib.ppo_mask import MaskablePPO
from benchmark import get_benchmarking_data
from CustomEncoder import CustomCombinedExtractor
from env import MPSPEnv
from tqdm import tqdm
import numpy as np
import torch
import wandb
import gym
import sys

# --- Config ---
tags = ['tanh-nonlinearity', 'tetris']
wandb_run_path = 'rl-msps/PPO-SB3/whngow47'
train_again = True
config = {
    # Environment
    'ROWS': 10,
    'COLUMNS': 4,
    'N_PORTS': 10,
    # Model
    'PI_LAYER_SIZES': [64, 64],
    'VF_LAYER_SIZES': [64, 64],
    'EMBEDDING_DIM': 8,
    'ENCODING_SIZE': 32,
    # Training
    'TOTAL_TIMESTEPS': 9600000,
    '_ENT_COEF': 1e-5,
    '_LEARNING_RATE': 1e-3,
    '_N_EPOCHS': 3,
    '_NORMALIZE_ADVANTAGE': False,
    '_N_STEPS': 2048,
    '_GAMMA': 0.99,
}
# --------------

wandb.login(
    # Get key from command line, default to None
    key=sys.argv[2] if len(sys.argv) > 2 else None
)

env = make_vec_env(
    lambda: MPSPEnv(
        config['ROWS'],
        config['COLUMNS'],
        config['N_PORTS']
    ),
    # Take cores from command line, default to 8
    n_envs=int(sys.argv[1]) if len(sys.argv) > 1 else 8,
)

policy_kwargs = {
    'activation_fn': torch.nn.Tanh,
    'net_arch': [{
        'pi': config['PI_LAYER_SIZES'],
        'vf': config['VF_LAYER_SIZES']
    }],
    'features_extractor_class': CustomCombinedExtractor,
    'features_extractor_kwargs': {
        'vocab_size': config['N_PORTS'],
        'embedding_dim': config['EMBEDDING_DIM'],
        'n_ports': config['N_PORTS'],
        'encoding_size': config['ENCODING_SIZE']
    },
}
create_new_run = not wandb_run_path or train_again

if create_new_run:
    run = wandb.init(
        project="PPO-SB3",
        entity="rl-msps",
        sync_tensorboard=True,
        name=f"N{config['N_PORTS']}_R{config['ROWS']}_C{config['COLUMNS']}",
        config=config,
        # Use command line arguments, otherwise input()
        notes=sys.argv[3] if len(sys.argv) > 3 else input('Notes: '),
        monitor_gym=True,
        tags=tags
    )

if wandb_run_path:
    model_file = wandb.restore('model.zip', run_path=wandb_run_path)
    model = MaskablePPO.load(
        model_file.name,
        env=env
    )
    if train_again:
        model.learn(
            total_timesteps=config['TOTAL_TIMESTEPS'],
            callback=WandbCallback(
                model_save_path=f"models/{run.id}",
                model_save_freq=config['TOTAL_TIMESTEPS'] // 4,
            ),
            progress_bar=True,
        )
else:
    model = MaskablePPO(
        policy='MultiInputPolicy',
        env=env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs=policy_kwargs,
        ent_coef=config['_ENT_COEF'],
        learning_rate=config['_LEARNING_RATE'],
        n_epochs=config['_N_EPOCHS'],
        normalize_advantage=config['_NORMALIZE_ADVANTAGE'],
        n_steps=config['_N_STEPS'],
        gamma=config['_GAMMA'],
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

eval_rewards = []
# Negative because env returns negative reward for shifts
paper_rewards = [-e['paper_result'] for e in eval_data]
paper_seeds = [e['seed'] for e in eval_data]

for e in tqdm(eval_data, desc='Evaluating'):
    # Creating seperate env for evaluation
    env = MPSPEnv(
        config['ROWS'],
        config['COLUMNS'],
        config['N_PORTS']
    )
    env = gym.wrappers.RecordVideo(
        env, video_folder=f'video/N{config["N_PORTS"]}_R{config["ROWS"]}_C{config["COLUMNS"]}_S{e["seed"]}')

    total_reward = 0
    obs = env.reset(
        transportation_matrix=e['transportation_matrix']
    )

    done = False
    while not done:
        action_mask = env.action_masks()
        action, _ = model.predict(
            obs,
            action_masks=action_mask,
            deterministic=True  # Deterministic for evaluation
        )
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(obs_tensor)
        env.unwrapped.probs = distribution.distribution.probs
        env.unwrapped.prev_action = action
        env.unwrapped.action_mask = action_mask

        env.render()
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    eval_rewards.append(total_reward)

if create_new_run:
    eval = {
        'mean_reward': np.mean(eval_rewards),
        'mean_paper_reward': np.mean(paper_rewards),
        'rewards': eval_rewards,
        'paper_rewards': paper_rewards,
        'paper_seeds': paper_seeds
    }
    run.summary['evaluation_benchmark'] = eval

    run.finish()
