#!/bin/bash

# *****User input: change location for svs folder and output folder
SVS_INPUT_PATH=/gpfs/alpine/med108/proj-shared/seer-pdac-images/pdac-genomic-study
PATCH_PATH=/gpfs/alpine/med108/proj-shared/shahira/seer-pdac-images_patches/pdac-genomic-study
OUT_DIR=/gpfs/alpine/med108/proj-shared/shahira/seer-pdac-images_out/pdac-genomic-study
CODE_DIR=/gpfs/alpine/med108/proj-shared/shahira/quip_classification/u24_lymphocyte/
#LYM_PREDICTION_BATCH_SIZE=200
LYM_PREDICTION_BATCH_SIZE=350
#incep-mix
MODEL_CONFIG_FILENAME=config_incep-mix_test_ext.ini
HEATMAP_VERSION_NAME=lym_incep-mix_probability
BINARY_HEATMAP_VERSION_NAME=lym_incep-mix_binary
##vgg-mix
#MODEL_CONFIG_FILENAME=config_vgg-mix_test_ext.ini
#HEATMAP_VERSION_NAME=lym_vgg-mix_probability
#BINARY_HEATMAP_VERSION_NAME=lym_vgg-mix_binary
##incep-new-train-v2
#MODEL_CONFIG_FILENAME=config_tcga_incv4_b128_crop100_noBN_d75_mix_filtered_by_testset_plus_new3_test.ini
#HEATMAP_VERSION_NAME=lym_incep-new-train-v2_probability
#BINARY_HEATMAP_VERSION_NAME=lym_incep-new-train-v2_binary
#TIL_JOB_ID=pred_test
# end of user input
