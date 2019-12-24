#!/bin/bash

source ../../conf/variables.sh

matlab -nodisplay -singleCompThread -r "sample_super_patch_coords; exit;"
matlab -nodisplay -singleCompThread -r "extract_super_patches; exit;"

exit 0
