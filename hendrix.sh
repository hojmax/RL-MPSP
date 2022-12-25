!/bin/bash
# we run on the gpu partition and we allocate 2 titanx gpus
SBATCH -p gpu --gres=gpu:2
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
SBATCH --time=4:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
# $1 is your wandb api key as a terminal argument
python main.py 4 $1 1