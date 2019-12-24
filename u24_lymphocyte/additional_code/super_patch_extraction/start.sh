#!/bin/bash

source ../../conf/variables.sh

matlab -nodisplay -singleCompThread -r "distribution_8x8_sample; exit;"
matlab -nodisplay -singleCompThread -r "extract_super_patches; exit;"

exit 0
