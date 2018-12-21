# Step 1. Build an image, with your nvcr.io credentials
sudo -E singularity build --writable tf-os-py3.simg docker://nvcr.io/nvidia/tensorflow:18.11-py3

# Step 2. Go into the singularity, and install packages
sudo -E singularity shell --writable tf-os-py3.simg

# In the singularity image, do the following.
echo -e '#!/bin/bash\nls -alFA "$@"\nexit 0' > /bin/ll && chmod a+x /bin/ll && \
cd /tmp && apt update && apt install -y vim less wget nano ssh && \
python3 -m pip install tqdm numpy scipy six scikit-image opencv-python && \
apt install -y libsm6 libxext6 libgtk2.0-dev && \
apt install -y libxml2-dev libsqlite3-dev cmake* && \
wget http://downloads.sourceforge.net/lcms/lcms2-2.7.tar.gz && \
        tar -xzvf lcms2-2.7.tar.gz && \
        cd lcms2-2.7 && ./configure && \
        make -j4 && make install && \
        cd .. && rm -rf lcms2-2.7* && \
wget http://downloads.sourceforge.net/libpng/libpng-1.6.22.tar.xz && \
        tar -xvf libpng-1.6.22.tar.xz && \
        cd libpng-1.6.22 && ./configure && \
        make -j4 && make install && \
        cd .. && rm -rf libpng-1.6.22* && \
wget http://download.osgeo.org/libtiff/tiff-4.0.6.tar.gz && \
        tar -xzvf tiff-4.0.6.tar.gz && \
        cd tiff-4.0.6 && ./configure && \
        make -j4 && make install && \
        cd .. && rm -rf tiff-4.0.6* && \
wget http://downloads.sourceforge.net/openjpeg.mirror/openjpeg-2.1.0.tar.gz && \
        tar -xzvf openjpeg-2.1.0.tar.gz && \
        cd openjpeg-2.1.0 && mkdir build && \
        cd build && cmake ../ && \
        make -j4 && make install && \
        cd ../.. && rm -rf openjpeg-2.1.0* && \
apt install -y libcairo2-dev libgdk-pixbuf2.0-dev && \
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/:${PKG_CONFIG_PATH} && \
ldconfig && \
apt install -y libjpeg-dev && \
wget https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz && \
        tar -xzvf openslide-3.4.1.tar.gz && \
        cd openslide-3.4.1 && ./configure && \
        make -j8 && make install && \
        cd ../ && rm -rf openslide-3.4.1* && \
ldconfig && \
python3 -m pip install --no-deps openslide-python && \
rm /usr/bin/python && \
ln -s /usr/bin/python3 /usr/bin/python && \
ldconfig

# You can now exit
# You can install other packages as well

# Step 3. Repeat the exact step 2 again. Somehow openslide need to be installed twice.

