from env import MPSPEnv
import numpy as np


env = MPSPEnv(10, 4, 10)

obs = env.reset()

del obs['will_block']
del obs['loading_list_length']

values = list(obs.values())

values = [np.array(v).flatten() for v in values]

concat = np.concatenate(values, axis=0)


mpsp_config = {
    'ROWS': 10,
    'COLUMNS': 4,
    'N_PORTS': 10
}

bay_size = mpsp_config["ROWS"] * mpsp_config["COLUMNS"]
port_size = 1
loading_list_size = mpsp_config["N_PORTS"]*(mpsp_config["N_PORTS"]-1)

print(bay_size + port_size + loading_list_size)

# reshape concat to (1,1, length of concat)
concat = concat.reshape(1,1,-1)

print(concat.shape)