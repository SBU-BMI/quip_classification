#!/bin/bash

source ../../conf/variables.sh

SLIDES=${SVS_INPUT_PATH}
IMG_FOLDER=${MODIFIED_HEATMAPS_PATH}

EMPTY_PRED_HEATMAP=${HEATMAP_TXT_OUTPUT_FOLDER}

rm -rf patch_coordinates
mkdir patch_coordinates
for files in ${IMG_FOLDER}/*.png; do
    if [ ! -f ${files} ]; then
        continue;
    fi
    SVS=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}'`
    USER=`echo ${files} | awk -F'/' '{print $NF}' | awk -F '.' '{print $2}'`
    FULL_SVS_PATH=`ls -1 ${SLIDES}/${SVS}*.svs | head -n 1`
    if [ ! -f "${FULL_SVS_PATH}" ]; then
        FULL_SVS_PATH=`ls -1 ${SLIDES}/${SVS}*.tif | head -n 1`
    fi
    FULL_SVS_NAME=`echo ${FULL_SVS_PATH} | awk -F'/' '{print $NF}'`
    if [ ! -f "${FULL_SVS_PATH}" ]; then
        echo image ${FULL_SVS_PATH} does not exist
        continue;
    fi

    WIDTH=` openslide-show-properties ${FULL_SVS_PATH} | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${FULL_SVS_PATH} | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`
    MPP=`openslide-show-properties ${FULL_SVS_PATH} | grep "aperio.MPP" | awk '{print substr($2,2,length($2)-2)}'`
    if [ "${WIDTH}" == "" ]; then
        echo Dimension of image ${FULL_SVS_PATH} is unknown
        continue
    fi

    matlab -nodisplay -singleCompThread -r \
    "get_patch_coordinates('${files}', '${SVS}', '${USER}', '${IMG}', ${WIDTH}, ${HEIGHT}, ${MPP}); exit;" \
    </dev/null
done

exit 0
