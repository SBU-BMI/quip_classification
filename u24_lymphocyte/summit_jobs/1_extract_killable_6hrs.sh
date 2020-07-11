#!/bin/bash


NUM_RUNS=$1
CLEAN_STARTED_FILES=$2
source 0_user_input.sh
#rm ${CODE_DIR}/data/log/log.color.txt
#rm ${CODE_DIR}/data/log/log.cnn.txt
#rm ${CODE_DIR}/data/log/log.prediction.txt
if [ $CLEAN_STARTED_FILES == 1 ]; then
  if [ -d ${PATCH_PATH} ]; then
    rm ${PATCH_PATH}/*/til_extract_started.txt
  fi
fi

for (( i=1; i<=${NUM_RUNS}; i++ ))
do
	TIL_JOB_ID=til_extract_k6_${i}
    TMP="tmp.lsf"
    echo "#!/bin/bash -x" > ${TMP}
    echo "#BSUB -P med108" >> ${TMP}
    echo "#BSUB -J ${TIL_JOB_ID}" >> ${TMP}
    echo "#BSUB -o ./logs/log.${TIL_JOB_ID}.o%J" >> ${TMP}
    echo "#BSUB -e ./logs/log.${TIL_JOB_ID}.e%J" >> ${TMP}
    echo "#BSUB -W 6:00" >> ${TMP}
    echo "#BSUB -B" >> ${TMP}
    echo "#BSUB -alloc_flags \"smt4\"" >> ${TMP}
    echo "#BSUB -nnodes 1" >> ${TMP}
    echo "#BSUB -q killable" >> ${TMP}
    echo "source ./utils/0_activate_environment.sh" >> ${TMP}
    echo "source 0_user_input.sh" >> ${TMP}
    echo "cd \${CODE_DIR}/patch_extraction" >> ${TMP}
    echo "jsrun -n 1 -a 1 -c 4 -g 0 -l GPU-CPU -b rs  --env TIL_JOB_ID=${TIL_JOB_ID}   --env SVS_INPUT_PATH=${SVS_INPUT_PATH}  --env PATCH_PATH=${PATCH_PATH}  --env OUT_DIR=${OUT_DIR}  --env CODE_DIR=${CODE_DIR}  --env LYM_PREDICTION_BATCH_SIZE=${LYM_PREDICTION_BATCH_SIZE}  --env MODEL_CONFIG_FILENAME=${MODEL_CONFIG_FILENAME}  --env HEATMAP_VERSION_NAME=${HEATMAP_VERSION_NAME}  --env BINARY_HEATMAP_VERSION_NAME=${BINARY_HEATMAP_VERSION_NAME}  bash ./start_summit.sh" >> ${TMP}

    bsub ${TMP}
    rm -f ${TMP}
done
