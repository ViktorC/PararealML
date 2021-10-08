#!/bin/bash

#SBATCH --job-name=vcsomor-parareal-ml
#SBATCH --time=08:00:0
#SBATCH --partition=gpu-cascade
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

#SBATCH --account=tc028-vcsomor

module load mpt
module load nvidia/cuda-11.2
module load nvidia/cudnn/8.2.1-cuda-11.2
module load nvidia/mathlibs-11.2
module load anaconda/python3

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=1

source activate pararealml

srun python3 -m examples.population_growth_parareal_ml
