FROM ubuntu:focal

LABEL Description="Ubuntu 20.04 environment with HDF5 1.14.1"

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
        python3 \ 
        python3-pip \
        python3-dev \
        python3-setuptools \
        gcc \
        g++ \
        libopenmpi-dev \
        software-properties-common \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' \
    && apt-get update \
    && apt-get install cmake -y \
    && pip3 install psutil \
    && wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_1.tar.gz \
    && tar zxvf hdf5-1_14_1.tar.gz \
    && mv hdf5-hdf5-1_14_1 hdf5 \
    && cd hdf5 \
    && ./autogen.sh \
    && CC=mpicc ./configure --prefix=/opt/hdf5 --enable-parallel --enable-threadsafe --enable-unsupported \
    && make -j 8 \
    && make install \
	&& cd .. \
	&& export PATH=/opt/hdf5/bin:$PATH \
	&& export LD_LIBRARY_PATH=/opt/hdf5/lib:$LD_LIBRARY_PATH \
	&& export HDF5_HOME=/opt/hdf5 \
	&& wget https://github.com/szcompressor/SZ3.git \
	&& cd SZ3 \
	&& mkdir build installer \
	&& export HDF5_HOME=/opt/hdf5 \
	&& export PATH=/opt/hdf5/bin:$PATH \
	&& cd build \
	&& cmake -DCMAKE_INSTALL_PREFIX=/opt/SZ3/installer -DBUILD_H5Z_FILTER=ON .. \
	&& make \
	&& make install \
	&& cd ../.. \
	&& wget https://github.com/LLNL/zfp.git \
	&& cd zfp \
	&& mkdir build installer \
	&& cd build \
	&& cmake -DCMAKE_INSTALL_PREFIX=/opt/zfp/installer -DZFP_BIT_STREAM_WORD_SIZE=8 .. \
	&& make \
	&& make install \
	&& cd ../.. \
	&& wget https://github.com/LLNL/H5Z-ZFP.git \
	&& cd H5Z-ZFP \
	&& mkdir build installer \
	&& cd build \
	&& export HDF5_DIR=/opt/hdf5 \
	&& export ZFP_DIR=/opt/zfp/installer/lib64/cmake/zfp \
	&& cmake -DCMAKE_INSTALL_PREFIX=/opt/H5Z-ZFP/installer .. \
	&& make \
	&& make install \
	&& cd ../..
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoclean
