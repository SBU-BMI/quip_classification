#!/bin/bash

# Load modules
module purge
module load matlab
module load mongodb/3.2.0
module load jdk8/1.8.0_11
module load openslide/3.4.0
module load extlibs/1.0.0
module load ITK/4.6.1
module load cuda75
module load anaconda2/4.4.0
module load imagemagick/7.0.7
export PATH=/home/lehhou/git/bin/:${PATH}
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cm/shared/apps/anaconda2/current/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cm/shared/apps/cuda75/toolkit/7.5.18/lib64/"
export CUDA_HOME=/cm/shared/apps/cuda75
export LIBTIFF_CFLAGS="-I/cm/shared/apps/extlibs/include" 
export LIBTIFF_LIBS="-L/cm/shared/apps/extlibs/lib -ltiff" 

matlab -nodisplay -singleCompThread -r "sample_super_patch_coords; exit;"
matlab -nodisplay -singleCompThread -r "extract_super_patches; exit;"

exit 0
