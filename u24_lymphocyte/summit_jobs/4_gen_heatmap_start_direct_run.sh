#!/bin/bash


#NUM_RUNS=$1
source 0_user_input.sh

TIL_JOB_ID=til_gen_heatmap_prepare
source ./utils/0_activate_environment.sh
source 0_user_input.sh
cd ${CODE_DIR}/heatmap_gen
bash ./start_summit.sh

