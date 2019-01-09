#!/bin/bash

# If you want to run 1 downloading thread:
# bash ${this_file} 0 1
# If you want to run 3 parallel downloading threads:
# bash ${this_file} 0 3
# bash ${this_file} 1 3
# bash ${this_file} 2 3

# What is my parallelization ID code
PAR_CODE=$1
# How many parallel downloading processes
PAR_MAX=$2

N=0
cat list_wsi_on_gcloud.txt | while read line; do
    N=$((N+1))
    if [ $((N%PAR_MAX)) -ne $PAR_CODE ]; then continue; fi

    GS_URL=`echo ${line} | awk '{print $1}'`
    CTYPE=`echo ${line} | awk '{print $2}'`
    SVS=`echo ${GS_URL} | awk -F'/' '{print $NF}'`

    mkdir -p ${CTYPE}
    if [ ! -f ${CTYPE}/${SVS} ]; then
        echo ${CTYPE}/${SVS}
        HTTP_URL=`echo ${GS_URL} | awk '{print "https://storage.googleapis.com/"substr($0, 6, length($0))}'`
        wget --quiet -O ${CTYPE}/${SVS} ${HTTP_URL}
    fi
done

exit 0
