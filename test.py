from env import MPSPEnv

env = MPSPEnv(3, 3, 3)
print(env.observation_space.sample())