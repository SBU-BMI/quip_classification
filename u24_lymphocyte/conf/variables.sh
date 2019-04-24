#!/bin/bash

# Variables
DEFAULT_OBJ=40
DEFAULT_MPP=0.25
CANCER_TYPE=quip
MONGODB_HOST=osprey.bmi.stonybrook.edu
MONGODB_PORT=27017
HEATMAP_VERSION=lym_vgg_mix

# Base directory
BASE_DIR=/root/u24_lymphocyte/

# The username you want to download heatmaps from
USERNAME=john.vanarnam@gmail.com
# The list of case_ids you want to download heaetmaps from
CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt

# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=${BASE_DIR}/data/heatmap_jsons
HEATMAP_TXT_OUTPUT_FOLDER=${BASE_DIR}/data/heatmap_txt
LOG_OUTPUT_FOLDER=${BASE_DIR}/data/log
SVS_INPUT_PATH=${BASE_DIR}/data/svs
PATCH_PATH=${BASE_DIR}/data/patches
PATCH_SAMPLING_LIST_PATH=${BASE_DIR}/data/patch_sample_list
RAW_MARKINGS_PATH=${BASE_DIR}/data/raw_marking_xy
MODIFIED_HEATMAPS_PATH=${BASE_DIR}/data/modified_heatmaps
TUMOR_HEATMAPS_PATH=${BASE_DIR}/data/tumor_labeled_heatmaps
TUMOR_GROUND_TRUTH=${BASE_DIR}/data/tumor_ground_truth_maps
TUMOR_IMAGES_TO_EXTRACT=${BASE_DIR}/data/tumor_images_to_extract
GRAYSCALE_HEATMAPS_PATH=${BASE_DIR}/data/grayscale_heatmaps
THRESHOLDED_HEATMAPS_PATH=${BASE_DIR}/data/thresholded_heatmaps
PATCH_FROM_HEATMAP_PATH=${BASE_DIR}/data/patches_from_heatmap
THRESHOLD_LIST=${BASE_DIR}/data/threshold_list/threshold_list.txt

CAE_TRAINING_DATA=${BASE_DIR}/data/training_data_cae
CAE_TRAINING_DEVICE=gpu0
CAE_MODEL_PATH=${BASE_DIR}/data/models_cae
LYM_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
LYM_CNN_TRAINING_DEVICE=gpu0
LYM_CNN_PRED_DEVICE=gpu0
#LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/data/models_cnn
#LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/config_vgg-mix_test_ext.ini
if [[ -n $MODEL_CONFIG_FILENAME ]]; then
  LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/${MODEL_CONFIG_FILENAME} ;
else
  LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/config_vgg-mix_test_ext.ini ;
fi
NEC_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
NEC_CNN_TRAINING_DEVICE=gpu0
NEC_CNN_PRED_DEVICE=gpu0
EXTERNAL_LYM_MODEL=1

