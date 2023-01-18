from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
from sb3_contrib.ppo_mask import MaskablePPO
from benchmark import get_benchmarking_data
from CustomEncoder import CustomCombinedExtractor
from env import MPSPEnv, NoRemoveWrapper, StrategicRemoveWrapper, RandomTrainingWrapper
from tqdm import tqdm
import numpy as np
import torch
import wandb
import gym
import sys
import argparse


parser = argparse.ArgumentParser(description="My parser")

parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb_key", type=str, default=None)
parser.add_argument("--show_progress", action="store_true")
parser.add_argument("--train_again", action="store_true")
parser.add_argument("--wandb_run_path", type=str, default=None)
parser.add_argument("--tags", type=str, default=None)
parser.add_argument("--notes", type=str, default=None)
parser.add_argument("--n_envs", type=int, default=1)
parser.add_argument("--remove_option", type=str, default="base")
parser.add_argument("--random_training", action="store_true")

args = parser.parse_args()

# --- Config ---
tags = args.tags.split(",") if args.tags else []
wandb_run_path = args.wandb_run_path
train_again = args.train_again
log_wandb = args.wandb
wandb_key = args.wandb_key
show_progress = args.show_progress
n_envs = args.n_envs
remove_option = args.remove_option
notes = args.notes
random_training = args.random_training


# Add automatic tags
tags.append(remove_option)


config = {
    # Environment
    "ROWS": 10,
    "COLUMNS": 4,
    "N_PORTS": 10,
    # Model
    "PI_LAYER_SIZES": [64, 64],
    "VF_LAYER_SIZES": [64, 64],
    "CONTAINER_EMBEDDING_SIZE": 8,
    "OUTPUT_HIDDEN": 256,
    "INTERNAL_HIDDEN": 32,
    # Training
    "TOTAL_TIMESTEPS": 3_000_000,
    "_ENT_COEF": 0,
    "_LEARNING_RATE": 1e-4,
    "_N_EPOCHS": 3,
    "_NORMALIZE_ADVANTAGE": True,
    "_N_STEPS": 2048,
    "_GAMMA": 0.99,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_vec_env(
    lambda: MPSPEnv(config["ROWS"], config["COLUMNS"], config["N_PORTS"]),
    # Take cores from command line, default to 8
    n_envs=n_envs,
)

policy_kwargs = {
    "activation_fn": torch.nn.Tanh,
    "net_arch": [{"pi": config["PI_LAYER_SIZES"], "vf": config["VF_LAYER_SIZES"]}],
    "features_extractor_class": CustomCombinedExtractor,
    "features_extractor_kwargs": {
        "n_ports": config["N_PORTS"],
        "container_embedding_size": config["CONTAINER_EMBEDDING_SIZE"],
        "internal_hidden": config["INTERNAL_HIDDEN"],
        "output_hidden": config["OUTPUT_HIDDEN"],
        "device": device,
    },
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
        notes=notes or input("Notes: "),
        monitor_gym=True,
        tags=tags,
    )


def make_env():
    return MPSPEnv(config["ROWS"], config["COLUMNS"], config["N_PORTS"])


def make_remove_option_env():
    if remove_option == "base":
        return make_env()
    elif remove_option == "no_remove":
        return NoRemoveWrapper(make_env())
    elif remove_option == "strategic_remove":
        return StrategicRemoveWrapper(make_env())


if random_training:
    env = make_vec_env(
        lambda: RandomTrainingWrapper(make_remove_option_env()),
        n_envs=n_envs,
    )
else:
    env = make_vec_env(
        lambda: make_remove_option_env(),
        n_envs=n_envs,
    )


if log_wandb:
    wandb.login(
        # Get key from command line, default to None
        key=wandb_key
    )


if wandb_run_path:
    model_file = wandb.restore("model.zip", run_path=wandb_run_path)
    model = MaskablePPO.load(model_file.name, env=env)
else:
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=0,
        tensorboard_log=f"runs/{run.id}" if create_new_run else None,
        policy_kwargs=policy_kwargs,
        ent_coef=config["_ENT_COEF"],
        learning_rate=config["_LEARNING_RATE"],
        n_epochs=config["_N_EPOCHS"],
        normalize_advantage=config["_NORMALIZE_ADVANTAGE"],
        n_steps=config["_N_STEPS"],
        gamma=config["_GAMMA"],
        device=device,
    )

if train_again or not wandb_run_path:
    model.learn(
        total_timesteps=config["TOTAL_TIMESTEPS"],
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            model_save_freq=config["TOTAL_TIMESTEPS"] // 4,
        )
        if create_new_run
        else None,
        progress_bar=True if show_progress else False,
    )

eval_data = get_benchmarking_data("rl-mpsp-benchmark/set_2")
eval_data = [
    e
    for e in eval_data
    if (
        e["R"] == config["ROWS"]
        and e["C"] == config["COLUMNS"]
        and e["N"] == config["N_PORTS"]
    )
]

eval_rewards = []
# Negative because env returns negative reward for shifts
paper_rewards = [-e["paper_result"] for e in eval_data]
paper_seeds = [e["seed"] for e in eval_data]

for e in tqdm(eval_data, desc="Evaluating"):
    # Creating seperate env for evaluation
    env = MPSPEnv(config["ROWS"], config["COLUMNS"], config["N_PORTS"])
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=f'video/N{config["N_PORTS"]}_R{config["ROWS"]}_C{config["COLUMNS"]}_S{e["seed"]}',
    )

    total_reward = 0
    obs = env.reset(transportation_matrix=e["transportation_matrix"])

    done = False
    while not done:
        action_mask = env.action_masks()
        action, _ = model.predict(
            obs,
            action_masks=action_mask,
            deterministic=True,  # Deterministic for evaluation
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
        "mean_reward": np.mean(eval_rewards),
        "mean_paper_reward": np.mean(paper_rewards),
        "rewards": eval_rewards,
        "paper_rewards": paper_rewards,
        "paper_seeds": paper_seeds,
    }
    run.summary["evaluation_benchmark"] = eval

    run.finish()
