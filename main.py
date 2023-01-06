from stable_baselines3.common.env_util import make_vec_env
from CustomEncoder import CustomCombinedExtractor
from wandb.integration.sb3 import WandbCallback
from sb3_contrib.ppo_mask import MaskablePPO
from benchmark import get_benchmarking_data
from env import MPSPEnv
from tqdm import tqdm
import numpy as np
import torch
import wandb
import gym
import sys


# --- Config ---
tags = ['state reduction', 'linear simple', 'C env', 'authentic matrices']
wandb_run_path = None
train_again = False
log_wandb = int(sys.argv[4]) if len(sys.argv) > 4 else True
show_progress = int(sys.argv[5]) if len(sys.argv) > 5 else True

config = {
    # Environment
    'ROWS': 10,
    'COLUMNS': 4,
    'N_PORTS': 10,
    # Model
    'PI_LAYER_SIZES': [32, 32],
    'VF_LAYER_SIZES': [32, 32],
    'CONTAINER_EMBEDDING_SIZE': 16,
    'INTERNAL_HIDDEN': 32,
    'OUTPUT_HIDDEN': 64,
    # Training
    'TOTAL_TIMESTEPS': 100e6,
    '_ENT_COEF': 0,
    '_LEARNING_RATE': 1.5e-4,
    '_N_EPOCHS': 3,
    '_NORMALIZE_ADVANTAGE': True,
    '_N_STEPS': 2048,
    '_GAMMA': 0.99,
}
# --------------

wandb.login(
    # Get key from command line, default to None
    key=sys.argv[2] if len(sys.argv) > 2 else None
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

policy_kwargs = {
    'activation_fn': torch.nn.Tanh,
    'net_arch': [{
        'pi': config['PI_LAYER_SIZES'],
        'vf': config['VF_LAYER_SIZES']
    }],
    'features_extractor_class': CustomCombinedExtractor,
    'features_extractor_kwargs': {
        'n_ports':  config['N_PORTS'],
        'container_embedding_size': config['CONTAINER_EMBEDDING_SIZE'],
        'internal_hidden': config['INTERNAL_HIDDEN'],
        'output_hidden': config['OUTPUT_HIDDEN'],
    }
}
create_new_run = (not wandb_run_path or train_again) and log_wandb

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

# Take cores from command line, default to 8
n_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 16

base_env = make_vec_env(
    lambda: MPSPEnv(
        rows=config['ROWS'],
        columns=config['COLUMNS'],
        n_ports=config['N_PORTS'],
        remove_restrictions="remove_only_when_blocking"
    ),
    n_envs=n_envs,
)

if wandb_run_path:
    model_file = wandb.restore('model.zip', run_path=wandb_run_path)
    model = MaskablePPO.load(
        model_file.name,
        env=base_env,
    )
    if train_again:
        print('Fine-tuning...')
        model.learn(
            total_timesteps=config['TOTAL_TIMESTEPS'],
            callback=WandbCallback(
                model_save_path=f"models/{run.id}",
            ) if create_new_run else None,
            progress_bar=show_progress,
        )
else:
    model = MaskablePPO(
        policy='MultiInputPolicy',
        env=base_env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}" if create_new_run else None,
        policy_kwargs=policy_kwargs,
        ent_coef=config['_ENT_COEF'],
        learning_rate=config['_LEARNING_RATE'],
        n_epochs=config['_N_EPOCHS'],
        normalize_advantage=config['_NORMALIZE_ADVANTAGE'],
        n_steps=config['_N_STEPS'],
        gamma=config['_GAMMA'],
        device=device,
    )
    model.learn(
        total_timesteps=config['TOTAL_TIMESTEPS'],
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
        ) if create_new_run else None,
        progress_bar=show_progress,
    )

base_env.close()

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

env = MPSPEnv(
    rows=config['ROWS'],
    columns=config['COLUMNS'],
    n_ports=config['N_PORTS'],
    remove_restrictions="remove_only_when_blocking"
)
env = gym.wrappers.RecordVideo(
    env, video_folder=f'video/N{config["N_PORTS"]}_R{config["ROWS"]}_C{config["COLUMNS"]}_S{0}'
)

for e in tqdm(eval_data, desc='Evaluating'):
    total_reward = 0

    obs = env.reset(
        transportation_matrix=e['transportation_matrix'].astype(np.int32)
    )

    done = False
    while not done:
        mask = env.action_masks()
        action, _ = model.predict(
            obs,
            action_masks=mask,
            deterministic=True  # Deterministic for evaluation
        )
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(obs_tensor)
        env.unwrapped.probs = distribution.distribution.probs

        env.render()
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    eval_rewards.append(total_reward)

env.close()

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
