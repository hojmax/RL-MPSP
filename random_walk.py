from env import MPSPEnv
import numpy as np
import pandas as pd
import json

env = MPSPEnv(10, 4, 10)


def state_to_json(state):
    return {
        'bay_matrix': state['bay_matrix'].tolist(),
        'port': int(state['port'][0]),
        'transportation_matrix': state['transportation_matrix'].tolist(),
    }



data = []

for i in range(100):

    # Reset and add initial state
    state = env.reset()
    t = 0
    data.append({
        'initial_state': state_to_json(state),
        'observations': [],
    })

    t += 1
    done = False
    while not done:
        
        # take a random action
        action_mask = env.action_masks()
        action_p = action_mask / np.sum(action_mask)
        
        action = np.random.choice(np.arange(len(action_mask)), p=action_p)
        state, reward, done, _ = env.step(action)
        data[-1]['observations'].append({
            'state': state_to_json(state),
            'reward': reward,
            'done': done,
            'action': int(action),
        })
        t += 1

# Serializing json
json_object = json.dumps(data, indent=4)
 
# Writing to sample.json
with open("data.json", "w") as outfile:
    outfile.write(json_object)


    