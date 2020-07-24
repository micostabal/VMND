#!/bin/bash
#SBATCH --partition=full
#SBATCH --job-name=VMND
#SBATCH --output=log.out
python Experiments.py testbed1.txt