#!/bin/bash
#SBATCH -p GPU
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:volta32:4
#SBATCH -N 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shahira 
#SBATCH --output=testing_out.txt


#echo commands to stdout
set -x

source /etc/profile.d/modules.sh
module load singularity/2.5.1
cd /home/shahira/NNFramework

singularity exec --writable  --bind /pylon5/ac3uump --nv $SCRATCH/containers/tensorflow/tf-18.11-py3-w  python sa_runners/tf-classifier_runner.py $HOME/NNFramework/config/config_test.ini >> testxx.txt


exit 0