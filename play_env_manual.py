from env import MPSPEnv
import numpy as np

env = MPSPEnv(
    rows=10,
    columns=4,
    n_ports=10,
    remove_restrictions="remove_all"
)

test_matrix = np.array(
    [[0, 0, 6, 6, 2, 5, 8, 2, 11, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 2, 0, 3],
     [0, 0, 0, 0, 3, 0, 0, 2, 0, 1],
     [0, 0, 0, 0, 0, 0, 2, 1, 2, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
     [0, 0, 0, 0, 0, 0, 0, 11, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 13, 5],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 31],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.int32
)

env.reset(test_matrix)

print("Instructions:")
print("The command aN will add a container to column N")
print("The command rN will remove a container from column N")
print("The command 'q' will quit the game")
print("The command 'n' will fill the board a good bunch (without changing port)")
print("Press enter to continue...")
input()

env.print()
sum_reward = 0

while True:
    action = input("Enter an action: ")
    is_invalid = False

    try:
        if action == "q":
            env.close()
            break
        if action == "n":
            for i in range(env.C):
                while env.column_counts[i] < env.R and np.sum(env.column_counts) < env.R * env.C - 1:
                    state, reward, is_terminated, info = env.step(i)
                    sum_reward += reward
                    env.print()
            continue
        elif action[0] == "a":
            action = int(action[1:])
            assert action < env.C
            assert action >= 0
        elif action[0] == "r":
            action = int(action[1:]) + env.C
            assert action < 2 * env.C
            assert action >= env.C
        else:
            print("Invalid action")
            is_invalid = True
    except:
        print("Invalid action")
        is_invalid = True

    if not is_invalid:
        state, reward, is_terminated, info = env.step(action)
        sum_reward += reward
        env.print()
