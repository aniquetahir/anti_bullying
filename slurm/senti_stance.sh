#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -t 4-00:00      # time in d-hh:mm:ss
#SBATCH -p general
#SBATCH -q grp_corman      # partition
#SBATCH -G 4            # Number of GPUs
#SBATCH --mem=400G      # RAM
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

eval "$(conda shell.bash hook)"
conda activate alpaca
/bin/sh -c "lmql serve-model /data/huanliu/llama/hf_weights/llama2-7B --cuda --load_in_8bit True" &
sleep 120
#python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd /data/huanliu/artahir/compressed_code/anti_bullying
export PYTHONPATH=$(pwd)
python stance_sentiment/lmql_query.py "$@"
