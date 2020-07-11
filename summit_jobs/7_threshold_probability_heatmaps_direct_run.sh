#!/bin/bash


source 0_user_input.sh

TIL_JOB_ID=til_threshold_probability_heatmaps
source ./utils/0_activate_environment.sh
source 0_user_input.sh
cd ${CODE_DIR}/scripts
bash ./threshold_probability_heatmaps.sh

