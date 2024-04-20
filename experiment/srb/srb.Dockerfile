FROM ubuntu:18.04

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building
RUN apt-get update && \
    apt-get install -y git \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    libegl1-mesa-dev \
    libglu1-mesa-dev\
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libpng-dev \
    liblapack-dev \
    libblas-dev \
    ffmpeg gnuplot \
    texlive-font-utils \
    libtiff-dev \
    openexr \
    libopenexr-dev \
    libsuitesparse-dev \
    mesa-common-dev

RUN git clone https://github.com/fwilliams/surface-reconstruction-benchmark.git

WORKDIR /surface-reconstruction-benchmark

ADD bin /surface-reconstruction-benchmark/bin

WORKDIR /surface-reconstruction-benchmark/bin

RUN cmake .. -DCMAKE_BUILD_TYPE=Release

RUN make -j4
