#!/bin/bash

# Variables
DEFAULT_OBJ=40
DEFAULT_MPP=0.25
MONGODB_HOST=xyz
MONGODB_PORT=27017
CANCER_TYPE=all

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

# Base data and output directories
export BASE_DIR=/root/quip_classification
export TIL_DIR=${BASE_DIR}/u24_lymphocyte/
export DATA_DIR=/data
export OUT_DIR=${DATA_DIR}/output

# Prediction folders
# Paths of data, log, input, and output
export SVS_INPUT_PATH=${DATA_DIR}/svs
export JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons
export BINARY_JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons_binary
export HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
export BINARY_HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt_binary
export LOG_OUTPUT_FOLDER=${OUT_DIR}/log
export PATCH_PATH=${DATA_DIR}/patches
export OUT_FOLDERS="${JSON_OUTPUT_FOLDER} ${BINARY_JSON_OUTPUT_FOLDER} ${HEATMAP_TXT_OUTPUT_FOLDER} ${BINARY_HEATMAP_TXT_OUTPUT_FOLDER} ${LOG_OUTPUT_FOLDER} ${PATCH_PATH}"

# Trained model
if [[ -n $MODEL_CONFIG_FILENAME ]]; then
  export LYM_NECRO_CNN_MODEL_PATH=${TIL_DIR}/prediction/NNFramework_TF_models/${MODEL_CONFIG_FILENAME} ;
else
  export LYM_NECRO_CNN_MODEL_PATH=${TIL_DIR}/prediction/NNFramework_TF_models/config_vgg-mix_test_ext.ini ;
fi
if [[ -n $LYM_PREDICTION_BATCH_SIZE ]]; then
  export LYM_NECRO_CNN_MODEL_PATH=${TIL_DIR}/prediction/NNFramework_TF_models/${MODEL_CONFIG_FILENAME} ;
else
  export LYM_NECRO_CNN_MODEL_PATH=${TIL_DIR}/prediction/NNFramework_TF_models/config_vgg-mix_test_ext.ini ;
fi

# Training folders
# The list of case_ids you want to download heaetmaps from
export CASE_LIST=${DATA_DIR}/raw_marking_to_download_case_list/case_list.txt
export PATCH_SAMPLING_LIST_PATH=${DATA_DIR}/patch_sample_list
export RAW_MARKINGS_PATH=${DATA_DIR}/raw_marking_xy
export MODIFIED_HEATMAPS_PATH=${DATA_DIR}/modified_heatmaps
export TUMOR_HEATMAPS_PATH=${DATA_DIR}/tumor_labeled_heatmaps
export TUMOR_GROUND_TRUTH=${DATA_DIR}/tumor_ground_truth_maps
export TUMOR_IMAGES_TO_EXTRACT=${DATA_DIR}/tumor_images_to_extract
export GRAYSCALE_HEATMAPS_PATH=${DATA_DIR}/grayscale_heatmaps
export THRESHOLDED_HEATMAPS_PATH=${DATA_DIR}/rates-cancertype-all-auto
export PATCH_FROM_HEATMAP_PATH=${DATA_DIR}/patches_from_heatmap
export THRESHOLD_LIST=${DATA_DIR}/threshold_list/threshold_list.txt

export CAE_TRAINING_DATA=${DATA_DIR}/training_data_cae
export CAE_TRAINING_DEVICE=gpu0
export CAE_MODEL_PATH=${DATA_DIR}/models_cae
export LYM_CNN_TRAINING_DATA=${DATA_DIR}/training_data_cnn
export LYM_CNN_TRAINING_DEVICE=gpu0
export LYM_CNN_PRED_DEVICE=gpu0
export NEC_CNN_TRAINING_DATA=${DATA_DIR}/training_data_cnn
export NEC_CNN_TRAINING_DEVICE=gpu0
export NEC_CNN_PRED_DEVICE=gpu0
export EXTERNAL_LYM_MODEL=1

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	export LYM_CNN_TRAINING_DEVICE=0
	export LYM_CNN_PRED_DEVICE=0
else
	export LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICES}
	export LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
fi

