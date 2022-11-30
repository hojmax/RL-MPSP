import sys
sys.path.insert(0, '/Users/christianjensen/Library/CloudStorage/OneDrive-UniversityofCopenhagen/2 - Projects/Combinatorial Optimization/RL-MPSP')


from env import MPSPEnv


def test_reset():
    env = MPSPEnv(2, 2, 2)
    env.reset()
    assert env.bay_matrix.shape == (2, 2)


def test_step():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed = 0)
    env.step(0) # Place container in column 0
    assert env.bay_matrix[1, 0] == 1

def test_print():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed = 0)
    env.print()

def test_get_masks():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed = 0)
    mask = env._get_masks()
    assert (mask['add_mask'] == [True, True]).all()
    assert (mask['remove_mask'] == [False, False]).all()

def test_add_container():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed = 0)
    env._add_container(1, 0)
    assert env.bay_matrix[1, 0] == 1
    assert env.transportation_matrix[0, 1] == 3



def test_remove_container():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed = 0)
    env._add_container(1, 0)
    env._remove_container(1, 0)
    assert env.bay_matrix[1, 0] == 0


def test_get_last_destination_container():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed = 0)
    env._add_container(1, 0)
    assert env._get_last_destination_container() == 1

def test_get_observation():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed = 0)
    env._add_container(1, 0)
    observation = env._get_observation()
    assert observation[0].shape == (2, 2)
    assert observation[1].shape == (2, 2)







