from env import MPSPEnv
import numpy as np

# Q-learning with a dictionary

config = {
    'ROWS': 10,
    'COLUMNS': 4,
    'N_PORTS': 10,
    'REMOVE_RESTRICTIONS': "no_remove",
}

state_to_qvalues = {}

env = MPSPEnv(
    rows=config['ROWS'],
    columns=config['COLUMNS'],
    n_ports=config['N_PORTS'],
    remove_restrictions=config['REMOVE_RESTRICTIONS']
)

default_qvalues = np.concatenate(
    (-np.ones(env.C) / 100, -np.ones(env.C))
)

# state_to_qvalues[state.tobytes()] = default_qvalues.copy()

n_games = 100000
gamma = 0.9
alpha = 0.001
min_prob = 0.05
t = 0

# Implement Q-learning here
for i in range(n_games):
    obs = env.reset(seed=0)
    state = np.concatenate(
        (obs['bay_matrix'].flatten(), obs['transportation_matrix'])
    ).tobytes()
    done = False
    delta_q = 0
    while not done:
        t += 1
        if state not in state_to_qvalues:
            state_to_qvalues[state] = default_qvalues.copy()
        qvalues = state_to_qvalues[state]
        # print(qvalues)
        mask = env.action_masks()
        if np.random.random() < max(np.exp(-i * alpha), min_prob):
            action = np.random.choice(
                np.arange(env.action_space.n), p=mask / np.sum(mask))
        else:
            q_max = np.abs(qvalues).max()
            # Arbitrary value (3), just needs to be larger than 2
            action = (
                qvalues - 3 * q_max * (1 - mask)
            ).argmax()

        obs, reward, done, _ = env.step(action)
        next_state = np.concatenate(
            (obs['bay_matrix'].flatten(), obs['transportation_matrix'])
        ).tobytes()
        if next_state not in state_to_qvalues:
            state_to_qvalues[next_state] = default_qvalues.copy()
        new_mask = env.action_masks()
        next_qvalues = state_to_qvalues[next_state]
        learning_rate = 1 / (1 + t / 10000)
        correction = (reward + gamma * np.max(next_qvalues[mask]) -
                      qvalues[action]) * learning_rate

        state_to_qvalues[state][action] = qvalues[action] + correction
        delta_q += np.abs(correction)
        state = next_state
    print('Reward:', env.state.contents.sum_reward,
          'Game:', i)
    print('N States', len(state_to_qvalues))
    print('Delta Q', delta_q)
    print('Prob', max(np.exp(-i * alpha), min_prob))
    print('Learning Rate', learning_rate)

env.close()
