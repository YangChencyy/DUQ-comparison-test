#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=DUQ_fmnist
#SBATCH --nodes=2
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --mail-user=rivachen@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=standard
#SBATCH --output=/home/rivachen/DUQ-comparison-test/fmnist_results_95.log

# module purge
# conda init bash
source activate GP

python DUQ.py 'FashionMNIST' 128 128  