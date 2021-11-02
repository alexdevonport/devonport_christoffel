#!/bin/bash
#SBATCH --job-name=christoffel
#SBATCH --account=fc_control
#SBATCH --partition=savio2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00 
#SBATCH --output=cfun-pacbayes-nys-log.txt
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alex_devonport@berkeley.edu
## Command(s) to run:
module load python  # for python 3
python3 experiments-savio.py