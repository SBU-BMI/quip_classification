#!/bin/bash
#SBATCH -p RM
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node 2
#SBATCH -N 4

set -x

cd /pylon5/ac3uump/lhou/
source /etc/profile.d/modules.sh
module load singularity

singularity exec tf-os-py3.simg python extract_process.py

exit 0
