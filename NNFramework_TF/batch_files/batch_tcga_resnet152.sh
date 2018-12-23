#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:volta16:4
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shahira 

#echo commands to stdout
set -x

source /etc/profile.d/modules.sh
module load singularity/2.5.1
cd /home/shahira/NNFramework

singularity exec --writable  --bind $SCRATCH --nv $SCRATCH/containers/tensorflow/tf-18.11-py3-w  python sa_runners/tf-classifier_runner.py $HOME/NNFramework/config/config_tcga_resnet-152.ini

exit 0