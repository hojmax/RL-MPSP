import sys
sys.path.insert(0, '/Users/axelhojmark/Desktop/RL-MPSP')
from env import MPSPEnv
import numpy as np

env = MPSPEnv(2, 2, 2)
env.reset(seed=0)
env.print()


def test_reset():
    env = MPSPEnv(2, 2, 2)
    env.reset()
    assert env.bay_matrix.shape == (2, 2)


def test_step():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed=0)
    env.step(0)  # Place container in column 0
    assert env.bay_matrix[1, 0] == 1


def test_print():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed=0)
    env.print()


def test_get_masks():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed=0)
    mask = env.action_masks()
    assert (mask == [True, True, False, False]).all()


def test_add_container():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed=0)
    env._add_container(1, 0)
    assert env.bay_matrix[1, 0] == 1
    assert env.transportation_matrix[0, 1] == 3


def test_remove_container():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed=0)
    env._add_container(1, 0)
    env._remove_container(1, 0)
    assert env.bay_matrix[1, 0] == 0


def test_get_last_destination_container():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed=0)
    env._add_container(1, 0)
    assert env._get_last_destination_container() == 1


def test_get_observation():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed=0)
    env._add_container(1, 0)
    observation = env._get_observation()
    assert observation['bay_matrix'].shape == (2, 2)
    assert observation['transportation_matrix'].shape == (2, 2)


def test_offload_containers1():
    env = MPSPEnv(2, 2, 2)
    env.reset(seed=0)
    env.port = 1
    env.bay_matrix = np.array([
        [1, 0],
        [1, 0]
    ])
    blocking_containers = env._offload_containers()
    expected = np.array([
        [0, 0],
        [0, 0]
    ])
    assert blocking_containers == 0
    assert (env.bay_matrix == expected).all()


def test_offload_containers2():
    env = MPSPEnv(3, 3, 4)
    env.reset(
        transportation_matrix=np.array([
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
    )
    env.port = 1
    env.bay_matrix = np.array([
        [1, 0, 3],
        [2, 1, 3],
        [1, 3, 1]
    ])
    blocking_containers = env._offload_containers()
    expected_bay_matrix = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 3, 0]
    ])
    expected_transportation_matrix = np.array([
        [0, 0, 0, 0],
        [0, 0, 2, 3],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    expected_blocking_containers = 3
    assert blocking_containers == expected_blocking_containers
    assert (env.bay_matrix == expected_bay_matrix).all()
    assert (env.transportation_matrix == expected_transportation_matrix).all()