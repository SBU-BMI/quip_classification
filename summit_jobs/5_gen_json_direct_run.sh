#!/bin/bash


source 0_user_input.sh
TIL_JOB_ID=til_json
source ./utils/0_activate_environment.sh
source 0_user_input.sh
cd ${CODE_DIR}/heatmap_gen
bash ./gen_all_json_summit.sh
