#!/bin/bash

source ../conf/variables.sh

FN=$1
LYM_FOLDER=${INTERMEDIATE_FOLDER}/patch-level-lym/
OUT_FOLDER=${INTERMEDIATE_FOLDER}/patch-level-merged/

awk '{
    print $1, $2, $3, 0.0;
}' ${LYM_FOLDER}/${FN} > ${OUT_FOLDER}/${FN}

exit 0

