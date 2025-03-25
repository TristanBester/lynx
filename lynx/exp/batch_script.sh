#!/bin/bash
#SBATCH --partition=bigbatch 
#SBATCH --nodes=1
#SBATCH --time=72:00:00 
#SBATCH --job-name=snake-exp
#SBATCH --output=/home-mscluster/tbester/lynx/slurm_logs/out/out_file.%N.%j.out
#SBATCH --error=/home-mscluster/tbester/lynx/slurm_logs/err/error_file.%N.%j.err
#SBATCH --array=0-99%5 
#SBATCH --exclusive # Use the whole node exclusively


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

# FIXME: The relationship here between seed count and agents per node makes no sense
# Run multiple W&B agents in parallel
uv run lynx/agents/dqn/exp.py experiment.seed=$(( ${SLURM_ARRAY_TASK_ID} + 0 )) &
wait;