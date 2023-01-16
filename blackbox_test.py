from env import MPSPEnv
import numpy as np
from tqdm import tqdm


env = MPSPEnv(
    rows=10,
    columns=4,
    n_ports=10,
    remove_restrictions="remove_all",
    transportation_type="mixed"
)

env.reset()
runs = 1000000

last_ten_states = []

for i in tqdm(range(runs)):
    for i in range(env.C):
        if env.column_counts[i] < env.R:
            state, reward, is_terminated, info = env.step(i)

            last_ten_states.append(env.print(return_string=True))
            if len(last_ten_states) > 10:
                last_ten_states.pop(0)

            # Broad check for invalid states
            suspect = np.any(env.loading_list < 0) or env.state.contents.loading_list_length < 0 or \
                env.state.contents.port != 0 or np.all(env.mask == 0) or np.any(env.transportation_matrix < 0) or \
                np.any(env.transportation_matrix > 40) or \
                np.any(env.bay_matrix < 0) or np.any(env.bay_matrix >= 10) or \
                np.any(env.column_counts < 0) or np.any(env.column_counts > 10) or \
                np.any(env.containers_per_port < 0) or np.any(
                env.containers_per_port > 40)

            if suspect:
                for state in last_ten_states:
                    print(state)
                print("Suspect state found!")
                exit()

            if is_terminated:
                env.reset()
