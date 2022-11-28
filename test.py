from env import MPSPEnv

env = MPSPEnv(3, 3, 3)
print(env.observation_space.sample())
for i in range(10):
    print(env._get_short_distance_transportation_matrix(5))
