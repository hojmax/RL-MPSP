from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
from sb3_contrib.ppo_mask import MaskablePPO
from benchmark import get_benchmarking_data
from CustomEncoder import CustomCombinedExtractor
from env import MPSPEnv
import numpy as np
import torch
import wandb
import sys

wandb.login(
    # Get key from command line, default to None
    key=sys.argv[2] if len(sys.argv) > 2 else None
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config,
                    monitor_gym=True,
                    sync_tensorboard=True):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

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
                'n_ports': config['N_PORTS'],
                'container_embedding_size': config['CONTAINER_EMBEDDING_SIZE'],
                'internal_hidden': config['INTERNAL_HIDDEN'],
                'output_hidden': config['OUTPUT_HIDDEN'],
                'device': device
            },
        }

        model = MaskablePPO(
            policy='MultiInputPolicy',
            env=env,
            verbose=0,
            tensorboard_log=f"runs/{wandb.run.id}",
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
                model_save_path=f"runs/{wandb.run.id}",
            ),
            # progress_bar=True,
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

        for e in eval_data:
            # Creating seperate env for evaluation
            env = MPSPEnv(
                config['ROWS'],
                config['COLUMNS'],
                config['N_PORTS']
            )

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
        wandb.summary['evaluation_benchmark'] = eval


sweep_id = 'rl-msps/PPO-SB3/ggtbs7ys'

wandb.agent(sweep_id, train)
