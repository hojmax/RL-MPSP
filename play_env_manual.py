from env import MPSPEnv

env = MPSPEnv(
    rows=10,
    columns=4,
    n_ports=10
)
env.reset()

# Play the environment using console input
# The command aN will add a container to column N
# The command rN will remove a container from column N
# The command q will quit the game

env.print()
print()
sum_reward = 0

while True:
    action = input("Enter an action: ")
    is_invalid = False

    try:
        if action == "q":
            break
        elif action[0] == "a":
            action = int(action[1:])
        elif action[0] == "r":
            action = int(action[1:]) + env.C
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
        print("Reward: {}".format(reward))
        print("Sum reward: {}".format(sum_reward))
        print("Is terminated: {}".format(is_terminated))
        print("Mask: {}".format(info["mask"]))
        print("Will block: {}".format(state["will_block"]))
        print()
