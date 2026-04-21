FROM ubuntu:24.04

LABEL Description="Ubuntu 24.04 environment with HDF5 2.0.0"

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
        software-properties-common

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ noble main' \
    && apt-get update \
    && apt-get install cmake -y \
    && pip3 install psutil

RUN wget https://github.com/HDFGroup/hdf5/archive/refs/tags/2.0.0.tar.gz \
    && tar zxvf 2.0.0.tar.gz \
    && mv hdf5-2.0.0 hdf5 \
    && cd hdf5 \
    && mkdir build \
    && cd build \
    && CC=mpicc cmake -DCMAKE_INSTALL_PREFIX=/opt/hdf5 -DHDF5_ENABLE_PARALLEL=ON -DHDF5_ALLOW_UNSUPPORTED=ON -DHDF5_ENABLE_THREADSAFE=ON .. \
    && make -j 8 \
    && make install

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoclean