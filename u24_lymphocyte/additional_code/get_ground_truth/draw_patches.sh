#!/bin/bash

source ../../conf/variables.sh

SLIDES=../../data/svs/
mkdir -p patches
PAR=30

N=0
for files in ./patch_coordinates/*.txt; do
    SVS=`echo ${files} | awk -F'/' '{print $NF}' | awk -F '.' '{print $1}'`
    FULL_SVS_PATH=`ls -1 ${SLIDES}/${SVS}*.svs | head -n 1`
    if [ ! -f "${FULL_SVS_PATH}" ]; then
        FULL_SVS_PATH=`ls -1 ${SLIDES}/${SVS}*.tif | head -n 1`
    fi
    FULL_SVS_NAME=`echo ${FULL_SVS_PATH} | awk -F'/' '{print $NF}'`
    if [ ! -f "${FULL_SVS_PATH}" ]; then
        echo Image ${FULL_SVS_PATH} does not exist
        continue
    fi

    nohup python draw_patches.py ${files} ${FULL_SVS_PATH} &
    N=$((N+1))
    if [ $N -ge $PAR ]; then
        wait
        N=0
    fi
done

exit 0
