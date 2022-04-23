# FROM	nvcr.io/nvidia/tensorflow:18.11-py3
FROM tensorflow/tensorflow:1.13.1-gpu-py3
MAINTAINER quip_classification 

RUN 	apt-get -y update && \
	apt-get -y install python3-pip openslide-tools wget git && \
	pip install imageio==2.9.0 openslide-python==1.1.2 scikit-image==0.15.0 pymongo 

ENV	BASE_DIR="/quip_app/quip_classification"
ENV	PATH="./":$PATH

COPY	. ${BASE_DIR}/.

ENV     MODEL_VER="v1.0"
ENV	MODEL_URL="https://stonybrookmedicine.box.com/shared/static/bl15zu4lwb9cc7ltul15aa8kyrn7kh2d.zip"

RUN	cd ${BASE_DIR}/u24_lymphocyte/prediction/NNFramework_TF_models && \
	wget -v -O models.zip -L $MODEL_URL >/dev/null 2>&1 && \
    	unzip -o models.zip && rm -f models.zip && \
	bash ./update_path.sh && \
	chmod 0755 ${BASE_DIR}/u24_lymphocyte/scripts/*

WORKDIR ${BASE_DIR}/u24_lymphocyte/scripts

CMD ["/bin/bash"]
