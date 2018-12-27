#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:volta32:1
#SBATCH -N 1
#SBATCH --output=test_out.txt


#echo commands to stdout
set -x

source /etc/profile.d/modules.sh
module load singularity/2.5.1
cd /home/shahira/NNFramework

singularity exec --writable  --bind /pylon5/ac3uump --nv $SCRATCH/containers/tensorflow/tf-18.11-py3-w  python sa_runners/tf-classifier_runner.py $HOME/NNFramework/config_tcga.ini


exit 0