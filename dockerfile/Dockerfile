FROM	nvcr.io/nvidia/tensorflow:18.11-py3
MAINTAINER quip_classification 

ENTRYPOINT []

RUN 	apt-get -y update && \
	apt-get -y install python3-pip openslide-tools wget && \
	pip install openslide-python scikit-image pymongo 

WORKDIR /root

RUN	git clone https://github.com/SBU-BMI/quip_classification && \
	cd /root/quip_classification/u24_lymphocyte/prediction/NNFramework_TF_models && \
	wget -v -O models.zip -L \
		https://stonybrookmedicine.box.com/shared/static/bl15zu4lwb9cc7ltul15aa8kyrn7kh2d.zip >/dev/null 2>&1 && \
        unzip -o models.zip && rm -f models.zip && \
	chmod 0755 /root/quip_classification/u24_lymphocyte/scripts/*

ENV	BASE_DIR="/root/quip_classification/u24_lymphocyte"
ENV	PATH="./":$PATH
WORKDIR /root/quip_classification/u24_lymphocyte/scripts

CMD ["/bin/bash"]
