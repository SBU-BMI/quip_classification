#!/bin/bash


source 0_user_input.sh
#rm ${CODE_DIR}/data/log/log.color.txt
#rm ${CODE_DIR}/data/log/log.cnn.txt
#rm ${CODE_DIR}/data/log/log.prediction.txt
if [ -d ${PATCH_PATH} ]; then
  rm ${PATCH_PATH}/*/til_extract_started.txt
fi
