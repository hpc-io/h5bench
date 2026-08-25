FROM ubuntu:focal

LABEL Description="Ubuntu 20.04 environment with HDF5 1.14.1 (compression filters)"

ENV DEBIAN_FRONTEND=noninteractive 
ENV HDF5_LIBTOOL=/usr/bin/libtoolize

RUN apt-get update \
    && apt-get install -y \
        git \
        curl \
        wget \
        sudo \
        gpg \
        ca-certificates \
        m4 \
        autoconf \
        automake \
        libtool \
        pkg-config \
        cmake \
        libtool \
        zlib1g-dev \
        libaec0 \
        libaec-dev \ 
        python3 \ 
        python3-pip \
        python3-dev \
        python3-setuptools \
        gcc \
        g++ \
        libopenmpi-dev \
        software-properties-common \
    && ldconfig
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' \
    && apt-get update \
    && apt-get install cmake -y \
    && pip3 install psutil
RUN wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_1.tar.gz \
    && tar zxvf hdf5-1_14_1.tar.gz \
    && mv hdf5-hdf5-1_14_1 hdf5 \
    && cd hdf5 \
    && ./autogen.sh \
    && CC=mpicc CXX=mpicxx ./configure --prefix=/opt/hdf5 --enable-parallel --enable-cxx --enable-threadsafe --enable-shared --disable-static --enable-unsupported --with-zlib=yes --with-szlib=yes \
    && make -j 8 \
    && make install
RUN cd ~ \
	&& export PATH=/opt/hdf5/bin:$PATH \
	&& export LD_LIBRARY_PATH=/opt/hdf5/lib:$LD_LIBRARY_PATH \
	&& export HDF5_HOME=/opt/hdf5 \
	&& git clone https://github.com/szcompressor/SZ3 \
	&& cd SZ3 \
    && git checkout v3.1.8 \
	&& mkdir build \
	&& cd build \
	&& cmake -DCMAKE_INSTALL_PREFIX=/opt/SZ3 -DBUILD_H5Z_FILTER=ON .. \
	&& make \
	&& make install
RUN cd ~ \
	&& git clone https://github.com/LLNL/zfp \
	&& cd zfp \
	&& mkdir build \
	&& cd build \
	&& cmake -DCMAKE_INSTALL_PREFIX=/opt/zfp -DZFP_BIT_STREAM_WORD_SIZE=8 .. \
	&& make \
	&& make install
RUN cd ~ \
    && export HDF5_DIR=/opt/hdf5 \
    && export PATH=/opt/hdf5/bin:$PATH \
    && export LD_LIBRARY_PATH=/opt/hdf5/lib:$LD_LIBRARY_PATH \
    && export ZFP_DIR=/opt/zfp/lib/cmake/zfp \
	&& git clone https://github.com/LLNL/H5Z-ZFP \
	&& cd H5Z-ZFP \
	&& mkdir build \
	&& cd build \
	&& CC=mpicc cmake -DCMAKE_INSTALL_PREFIX=/opt/H5Z-ZFP -DFORTRAN_INTERFACE=OFF .. \
	&& make \
	&& make install
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoclean
