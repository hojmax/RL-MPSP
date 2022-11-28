from env import MPSPEnv

env = MPSPEnv(3, 3, 3)
env.reset()

# Play the environment using console input
# The command aN will add a container to column N
# The command rN will remove a container from column N
# The command q will quit the game

env.print()
print()

while True:
    action = input("Enter an action: ")
    is_invalid = False

    if action == "q":
        break
    elif action[0] == "a":
        action = int(action[1:])
    elif action[0] == "r":
        action = int(action[1:]) + env.C
    else:
        print("Invalid action")
        is_invalid = True

    if not is_invalid:
        state, reward, is_terminated, info = env.step(action)
        env.print()
        print("Reward: {}".format(reward))
        print("Is terminated: {}".format(is_terminated))
        print("Add mask: {}".format(info["add_mask"]))
        print("Remove mask: {}".format(info["remove_mask"]))
        print()
