#!/bin/bash

source ../conf/variables.sh

rm -rf ${INTERMEDIATE_FOLDER}/json \
       ${INTERMEDIATE_FOLDER}/patch-level-lym \
       ${INTERMEDIATE_FOLDER}/patch-level-nec \
       ${INTERMEDIATE_FOLDER}/patch-level-color \
       ${INTERMEDIATE_FOLDER}/patch-level-merged

mkdir  ${INTERMEDIATE_FOLDER}/json \
       ${INTERMEDIATE_FOLDER}/patch-level-lym \
       ${INTERMEDIATE_FOLDER}/patch-level-nec \
       ${INTERMEDIATE_FOLDER}/patch-level-color \
       ${INTERMEDIATE_FOLDER}/patch-level-merged

# Copy heatmap files from lym and necrosis prediction models
# to patch-level/ and necrosis/ folders respectively.
bash cp_heatmaps_all.sh ${PATCH_PATH} &> ${LOG_OUTPUT_FOLDER}/log.cp_heatmaps_all.txt

# Combine patch-level and necrosis heatmaps into one heatmap.
# Also generate high-res and low-res version.
bash combine_lym_necrosis_all.sh &> ${LOG_OUTPUT_FOLDER}/log.combine_lym_necrosis_all.txt
cp -f ${INTERMEDIATE_FOLDER}/patch-level-merged/* ${HEATMAP_TXT_OUTPUT_FOLDER}/
cp -f ${INTERMEDIATE_FOLDER}/patch-level-color/* ${HEATMAP_TXT_OUTPUT_FOLDER}/

# Generate meta and heatmap files for high-res and low-res heatmaps.
bash gen_all_json.sh &> ${LOG_OUTPUT_FOLDER}/log.gen_all_json.txt
cp -f ${INTERMEDIATE_FOLDER}/json/* ${JSON_OUTPUT_FOLDER}/

exit 0
