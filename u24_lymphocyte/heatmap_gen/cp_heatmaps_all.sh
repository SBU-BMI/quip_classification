#!/bin/bash

source ../conf/variables.sh

FOLDER=${1}

PRED_VERSION=patch-level-lym.txt
DIS_FOLDER=${OUT_DIR}/patch-level-lym/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp -f ${files} ${DIS_FOLDER}/${dis}
done

PRED_VERSION=patch-level-necrosis.txt
DIS_FOLDER=${OUT_DIR}/patch-level-nec/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp -f ${files} ${DIS_FOLDER}/${dis}
done

PRED_VERSION=patch-level-color.txt
DIS_FOLDER=${OUT_DIR}/patch-level-color/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "color-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp -f ${files} ${DIS_FOLDER}/${dis}
done

exit 0
