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
RUN	git clone https://github.com/SBU-BMI/u24_lymphocyte && \
	git clone https://github.com/SBU-BMI/quip_classification

CMD ["/bin/bash"]
