#!/bin/bash
#SBATCH --partition=full
#SBATCH --job-name=VMND
#SBATCH --output=log.out
python VMND/Instances/Experiments.py