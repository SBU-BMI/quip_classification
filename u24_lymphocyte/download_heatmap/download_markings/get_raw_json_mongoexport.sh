#!/bin/bash

mkdir -p raw_json

cat lists.txt | while read svs; do
    mongoexport \
      --collection=objects \
      --db=quip \
      --query='{"provenance.analysis.execution_id":"humanmark","provenance.image.case_id":"'${svs}'","properties.annotations.username":"dr.rajarsi.gupta@gmail.com"}' \
      --out=raw_json/${svs}.json
done

exit 0
