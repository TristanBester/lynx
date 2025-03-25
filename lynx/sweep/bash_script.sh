#!/bin/bash
mkdir -p /home-mscluster/tbester/lynx/slurm_logs/err 
mkdir -p /home-mscluster/tbester/lynx/slurm_logs/out 

# Setup the environment
cd /home-mscluster/tbester/lynx
uv sync 

# Copy batch scripts into project root 
cp /home-mscluster/tbester/lynx/lynx/sweep/batch_script.sh .

# Start the agents 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
sbatch batch_script.sh 
