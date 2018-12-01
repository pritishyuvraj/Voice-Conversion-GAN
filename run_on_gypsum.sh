#!/bin/bash
#
#BATCH --mem=100000
#SBATCH --job-name=1-gpu-bidaf-pytorch
#SBATCH --partition=m40-long
#SBATCH --output=bidaf-pytorch-%A.out
#SBATCH --error=bidaf-pytorch-%A.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pyuvraj@cs.umass.edu
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python/3.6.1
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44


# Training 
python -m train.py
