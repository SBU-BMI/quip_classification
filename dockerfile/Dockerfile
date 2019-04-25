FROM	nvcr.io/nvidia/tensorflow:18.11-py3
MAINTAINER test 

RUN 	apt-get -y update && \
	apt -y install python3-pip openslide-tools wget && \
	pip install openslide-python scikit-image pymongo 

WORKDIR /root

RUN	git clone https://github.com/SBU-BMI/quip_classification && \
	cd /root/quip_classification/u24_lymphocyte/prediction/NNFramework_TF_models && \
	wget -v -O models.zip -L https://stonybrookmedicine.box.com/shared/static/jr6t55nvyikow0nasudc0u40bion7rsg.zip && \
        unzip -o models.zip && rm -f models.zip && \
	cd /root/quip_classification/u24_lymphocyte/scripts && chmod 0755 *

ENV	BASE_DIR="/root/quip_classification/u24_lymphocyte"
ENV	PATH="./":$PATH
WORKDIR /root/quip_classification/u24_lymphocyte/scripts

CMD ["/bin/bash"]
