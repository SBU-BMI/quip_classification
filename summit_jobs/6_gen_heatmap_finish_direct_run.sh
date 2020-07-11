#!/bin/bash


source 0_user_input.sh

TIL_JOB_ID=til_gen_heatmap_finish
source ./utils/0_activate_environment.sh
source 0_user_input.sh
cd ${CODE_DIR}/heatmap_gen
bash ./finish_summit.sh

