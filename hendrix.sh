!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
SBATCH -p gpu
SBATCH --job-name=MyJob
#number of independent tasks we are going to start in this script
SBATCH --ntasks=1
#number of cpus we want to allocate for each program
SBATCH --cpus-per-task=4
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
SBATCH --time=2-00:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program

# $1 is your wandb api key as a terminal argument
python main.py 4 $1 1