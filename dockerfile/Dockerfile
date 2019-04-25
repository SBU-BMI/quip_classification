FROM	nvcr.io/nvidia/tensorflow:18.11-py3
MAINTAINER Test Person 

RUN 	apt-get -y update && \
	apt -y install python3-pip && \
	pip install openslide-python && \
	apt-get -y install openslide-tools && \
	pip install scikit-image && \
	pip install pymongo 

WORKDIR /root

# This will clone SBU-BMI/u24_lymphocyte and SBU-BMI/quip_classification 
# repositories to /root.
# We need to add steps to clone and install codes from SBU-BMI/quip_classification
# and to copy scripts to common locations (eg. /usr/bin, /usr/local/bin) for execution
RUN	git clone https://github.com/SBU-BMI/quip_classification

ENV        BASE_DIR=/root/quip_classification/u24_lymphocyte
#COPY       ../../TIL_models/NNFramework_TF_models/ root/quip_classification/u24_lymphocyte/prediction
ONBUILD    COPY   root/quip_classification/u24_lymphocyte/scripts/svs_2_heatmap.sh /usr/local/bin

# COPY     $BASE_DIR/scripts/patch_2_heatmap.sh  /usr/local/bin
# COPY     $BASE_DIR/scripts/threshold_probability_heatmaps.sh /usr/local/bin
# COPY     $BASE_DIR/scripts/cleanup_heatmap.sh /usr/local/bin


CMD ["/bin/bash"]
