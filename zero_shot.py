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

# --- Config ---
wandb_run_path = "rl-msps/PPO-SB3/2hdkxmf7"
remove_option = "strategic_remove"
max_iter = 1000
config = {
    # Environment
    "ROWS": 14,
    "COLUMNS": 14,
    "N_PORTS": 16,
}


# --- Environment ---
def make_env():
    return MPSPEnv(config["ROWS"], config["COLUMNS"], config["N_PORTS"])


def make_remove_option_env():
    if remove_option == "base":
        return make_env()
    elif remove_option == "no_remove":
        return NoRemoveWrapper(make_env())
    elif remove_option == "strategic_remove":
        return StrategicRemoveWrapper(make_env())


env = make_remove_option_env()


# --- Model ---
model_file = wandb.restore("model.zip", run_path=wandb_run_path)
model = MaskablePPO.load(model_file.name, env=env)


# --- Benchmarking ---
eval_data = get_benchmarking_data("rl-mpsp-benchmark/set_2")
eval_rewards = {}
# Negative because env returns negative reward for shifts

paper_rewards = {}
for e in eval_data:
    R, C, N = e["R"], e["C"], e["N"]
    if (R, C, N) not in paper_rewards:
        paper_rewards[(R, C, N)] = []
    paper_rewards[(R, C, N)].append(-e["paper_result"])


for e in tqdm(eval_data, desc="Evaluating"):
    R, C, N = e["R"], e["C"], e["N"]
    if (R, C, N) not in eval_rewards:
        eval_rewards[(R, C, N)] = []

    # Creating seperate env for evaluation
    total_reward = 0
    obs = env.reset(transportation_matrix=e["transportation_matrix"])

    done = False
    iter = 0
    while not done and iter < max_iter:
        action_mask = env.action_masks()
        action, _ = model.predict(
            obs,
            action_masks=action_mask,
            deterministic=True,  # Deterministic for evaluation
        )
        env.render()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        iter += 1

    eval_rewards[(R, C, N)].append(total_reward)


# --- Results ---
for (R, C, N), rewards in eval_rewards.items():
    print(f"R: {R}, C: {C}, N: {N}")
    print(f"Mean Reward: {np.mean(rewards)}")
    print(f"Mean Paper Reward: {np.mean(paper_rewards[(R, C, N)])}")
    print(f"Reward: {rewards}")
    print(f"Paper Reward: {paper_rewards[(R, C, N)]}")
    print()
