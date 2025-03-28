#!/bin/bash
mkdir -p /home-mscluster/tbester/lynx/slurm_logs/err 
mkdir -p /home-mscluster/tbester/lynx/slurm_logs/out 

# Setup the environment
cd /home-mscluster/tbester/lynx
uv sync 
uv add "jax[cuda12]"

# Change to the batch directory
cd /home-mscluster/tbester/lynx/.lynx/batch

# Start the agents 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
