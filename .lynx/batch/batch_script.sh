#!/bin/bash
# specify a partition
#SBATCH -p bigbatch 
# specify number of nodes
#SBATCH -N 1
# specify the wall clock time limit for the job hh:mm:ss
#SBATCH -t 72:00:00 
# specify the job name
#SBATCH -J snake-sweep
# specify the filename to be used for writing output
#SBATCH -o /home-mscluster/tbester/lynx/slurm_logs/out/out_file.%N.%j.out
# specify the filename for stderr
#SBATCH -e /home-mscluster/tbester/lynx/slurm_logs/err/error_file.%N.%j.err

# Get the hostname of the current machine
HOSTNAME=$(hostname)

# Check if the hostname contains the substring "login"
if [[ "$HOSTNAME" == *"login"* ]]; then
    echo "Error: This script cannot be run on a machine with 'login' in its hostname."
    exit 1
fi

# Change to the remote project directory
cd "/home-mscluster/tbester/lynx"
uv sync
source .venv/bin/activate

# Run multiple W&B agents in parallel
wandb agent tristanbester1/sweep-dqn-puzzle/b5grgq9w &
wait;