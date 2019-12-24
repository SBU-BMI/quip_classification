#!/bin/bash

source ../../conf/variables.sh

for fn in raw_json/*.json; do
    len=`cat ${fn} | wc -l`
    if [ ${len} -eq 0 ]; then
        continue;
    fi
    case_id=`echo ${fn} | awk '{print substr($0,1,length($0)-5)}' | awk -F'/' '{print $NF}'`

    cat ${fn} \
        | awk -F'\\{\\"_id\\":' '{for(i=2;i<=NF;++i){print "\"_id\":"$i}}' \
        | awk -f raw_data_formating.awk | sort -k 5 -n \
        | grep -P "rajarsi|john" > ${RAW_MARKINGS_PATH}/${case_id}__x__raj_mark.txt
    len=`cat ${RAW_MARKINGS_PATH}/${case_id}__x__raj_mark.txt | wc -l`
    if [ ${len} -eq 0 ]; then
        rm ${RAW_MARKINGS_PATH}/${case_id}__x__raj_mark.txt
    else
        echo 0.0 0.0 0.0 > ${RAW_MARKINGS_PATH}/${case_id}__x__raj_weight.txt
        echo ${case_id}__x__raj_mark.txt
    fi
done

exit 0
