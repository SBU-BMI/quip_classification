#!/bin/bash

# Variables
DEFAULT_OBJ=40
DEFAULT_MPP=0.25
CANCER_TYPE=quip
MONGODB_HOST=xyz
MONGODB_PORT=27017
# The username you want to download heatmaps from
USERNAME=xyz

if [[ -n $HEATMAP_VERSION_NAME ]]; then
	export HEATMAP_VERSION=$HEATMAP_VERSION_NAME ;
else
	export HEATMAP_VERSION=lym_vgg-mix ;
fi
if [[ -n $BINARY_HEATMAP_VERSION_NAME ]]; then
	export BINARY_HEATMAP_VERSION=$BINARY_HEATMAP_VERSION_NAME ;
else
	export BINARY_HEATMAP_VERSION=lym_vgg-mix_binary ;
fi
if [[ ! -n $LYM_PREDICTION_BATCH_SIZE ]]; then
   export LYM_PREDICTION_BATCH_SIZE=96;
fi

# Base directory
export BASE_DIR=/root/quip_classification/u24_lymphocyte/
export OUT_DIR=${BASE_DIR}/data/output

# The list of case_ids you want to download heaetmaps from
export CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt

# Paths of data, log, input, and output
export JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons
export BINARY_JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons_binary
export HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
export BINARY_HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt_binary
export LOG_OUTPUT_FOLDER=${OUT_DIR}/log
export INTERMEDIATE_FOLDER=${OUT_DIR}/intermediate

# Folders to input images and image patches to be created for prediction
export SVS_INPUT_PATH=${BASE_DIR}/data/svs
export PATCH_PATH=${BASE_DIR}/data/patches

export PATCH_SAMPLING_LIST_PATH=${BASE_DIR}/data/patch_sample_list
export RAW_MARKINGS_PATH=${BASE_DIR}/data/raw_marking_xy
export MODIFIED_HEATMAPS_PATH=${BASE_DIR}/data/modified_heatmaps
export TUMOR_HEATMAPS_PATH=${BASE_DIR}/data/tumor_labeled_heatmaps
export TUMOR_GROUND_TRUTH=${BASE_DIR}/data/tumor_ground_truth_maps
export TUMOR_IMAGES_TO_EXTRACT=${BASE_DIR}/data/tumor_images_to_extract
export GRAYSCALE_HEATMAPS_PATH=${BASE_DIR}/data/grayscale_heatmaps
export THRESHOLDED_HEATMAPS_PATH=${OUT_DIR}/rates-cancertype-all-auto
export PATCH_FROM_HEATMAP_PATH=${BASE_DIR}/data/patches_from_heatmap
export THRESHOLD_LIST=${BASE_DIR}/data/threshold_list/threshold_list.txt

export CAE_TRAINING_DATA=${BASE_DIR}/data/training_data_cae
export CAE_TRAINING_DEVICE=gpu0
export CAE_MODEL_PATH=${BASE_DIR}/data/models_cae
export LYM_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
export LYM_CNN_TRAINING_DEVICE=gpu0
export LYM_CNN_PRED_DEVICE=gpu0

if [[ -n $MODEL_CONFIG_FILENAME ]]; then
  export LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/${MODEL_CONFIG_FILENAME} ;
else
  export LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/config_vgg-mix_test_ext.ini ;
fi
if [[ -n $LYM_PREDICTION_BATCH_SIZE ]]; then
  export LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/${MODEL_CONFIG_FILENAME} ;
else
  export LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/prediction/NNFramework_TF_models/config_vgg-mix_test_ext.ini ;
fi

export NEC_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
export NEC_CNN_TRAINING_DEVICE=gpu0
export NEC_CNN_PRED_DEVICE=gpu0
export EXTERNAL_LYM_MODEL=1

# create missing output directories
if [ ! -d ${OUT_DIR} ]; then
  mkdir ${OUT_DIR} ;
fi

if [ ! -d ${JSON_OUTPUT_FOLDER} ]; then
  mkdir ${JSON_OUTPUT_FOLDER} ;
fi

if [ ! -d ${BINARY_JSON_OUTPUT_FOLDER} ]; then
  mkdir ${BINARY_JSON_OUTPUT_FOLDER} ;
fi

if [ ! -d ${HEATMAP_TXT_OUTPUT_FOLDER} ]; then
  mkdir ${HEATMAP_TXT_OUTPUT_FOLDER} ;
fi

if [ ! -d ${BINARY_HEATMAP_TXT_OUTPUT_FOLDER} ]; then
  mkdir ${BINARY_HEATMAP_TXT_OUTPUT_FOLDER} ;
fi

if [ ! -d ${LOG_OUTPUT_FOLDER} ]; then
  mkdir ${LOG_OUTPUT_FOLDER} ;
fi

if [ ! -d ${INTERMEDIATE_FOLDER} ]; then
  mkdir ${INTERMEDIATE_FOLDER} ;
fi

if [ ! -d ${PATCH_PATH} ]; then
  mkdir ${PATCH_PATH} ;
fi


