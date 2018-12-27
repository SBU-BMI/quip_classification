#!/bin/bash
#SBATCH -p GPU-AI
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:volta16:4
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shahira 
#SBATCH --output=tcga_inception-res_2class_adam_b16_CEloss_lr10-4_out.txt

#echo commands to stdout
set -x

source /etc/profile.d/modules.sh
module load singularity/2.5.1
cd /home/shahira/NNFramework

singularity exec --writable  --bind $SCRATCH --nv $SCRATCH/containers/tensorflow/tf-18.11-py3-w  python sa_runners/tf-classifier_runner.py $HOME/NNFramework/config_tcga.ini >> inc_res_out.txt

exit 0