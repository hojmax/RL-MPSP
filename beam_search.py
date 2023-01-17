from sb3_contrib.ppo_mask import MaskablePPO
from benchmark import get_benchmarking_data
from env import MPSPEnv
from tqdm import tqdm
import numpy as np
import wandb
import copy 
import gym


config = {
    # Environment
    'ROWS': 10,
    'COLUMNS': 4,
    'N_PORTS': 10,
    # Search parameters
    'N_BEAMS': 10,
    'MAX_STEPS': 200,
}

wandb_run_path = "rl-msps/PPO-SB3/2hdkxmf7"

model_file = wandb.restore('model.zip', run_path=wandb_run_path)
model = MaskablePPO.load(
    model_file.name
)

eval_data = get_benchmarking_data('rl-mpsp-benchmark/set_2')
eval_data = [
    e for e in eval_data if (
        e['R'] == config['ROWS'] and
        e['C'] == config['COLUMNS'] and
        e['N'] == config['N_PORTS']
    )
]

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
    initial_env = MPSPEnv(
        config['ROWS'],
        config['COLUMNS'],
        config['N_PORTS']
    )

    total_reward = 0
    obs = initial_env.reset(
        transportation_matrix=e['transportation_matrix']
    )

    # (log_prob, env, cumulative_reward, done, action)
    top_k = [(0, initial_env, 0, False, None)]

    all_done = False
    step = 0

    while not all_done and step < config['MAX_STEPS']:

        step += 1
    
        all = []
        for info in top_k:
            log_prob, env, cumulative_reward, done, action = info

            # Take action
            if action is not None and not done:
                obs, reward, done, _ = env.step(action)
                cumulative_reward += reward

            # Check if done
            if done:
                all.append((log_prob, env, cumulative_reward, done, action))
                continue

            # Predict next action probabilities
            action_mask = env.action_masks()
            action, _ = model.predict(
                obs,
                action_masks=action_mask,
                deterministic=True  # Deterministic for evaluation
            )
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            distribution = model.policy.get_distribution(obs_tensor)

            # Get log probabilities
            log_probs = distribution.distribution.probs.cpu().detach().numpy()[0]

            for a, p in enumerate(log_probs):
                # Append if not masked
                
                if action_mask[a] == 1:
                    all.append((log_prob + p, copy.deepcopy(env), cumulative_reward, done, a))


        # Sort by log probability
        all.sort(key=lambda x: x[0], reverse=True)

        # Keep top k
        top_k = all[:config['N_BEAMS']]

        # Check if all done
        all_done = True
        for _, _, _, done, _ in top_k:
            if not done:
                all_done = False
                break

    # Get best reward
    best_reward = -np.inf
    for _, _, cumulative_reward, _, _ in top_k:
        if cumulative_reward > best_reward:
            best_reward = cumulative_reward

    eval_rewards.append(best_reward)
        


eval = {
    'mean_reward': np.mean(eval_rewards),
    'mean_paper_reward': np.mean(paper_rewards),
    'rewards': eval_rewards,
    'paper_rewards': paper_rewards,
    'paper_seeds': paper_seeds
}

print(eval)