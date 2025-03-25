#!/bin/bash
mkdir -p /home-mscluster/tbester/lynx/slurm_logs/err 
mkdir -p /home-mscluster/tbester/lynx/slurm_logs/out 

# Setup the environment
cd /home-mscluster/tbester/lynx
uv sync 
uv add "jax[cuda12]"

# Copy batch scripts into project root 
cp /home-mscluster/tbester/lynx/lynx/exp/batch_script.sh .

# Start the agents 
sbatch batch_script.sh 0
