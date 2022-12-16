from env import MPSPEnv
from benchmark import get_benchmarking_data
import numpy as np
import pygame

env = MPSPEnv(10, 10, 10)

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

while True:
    env.render()



# done = False
# while not done:
#     env.render()
#     action, _ = model.predict(
#         obs,
#         action_masks=env.action_masks()
#     )
#     obs, reward, done, _ = env.step(action)
    