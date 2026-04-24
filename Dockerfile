FROM andrewseidl/nvidia-cuda:10.2-devel-ubuntu20.04

# update CUDA keyring

RUN apt-key del A4B469963BF863CC \
    && rm /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/cuda.list

RUN apt update \
    && apt install wget -y \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb

# Install dependencies

RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install git gcc-8 g++-8 gfortran qt5-default libglew-dev libgl1-mesa-dev libeigen3-dev libqglviewer-dev-qt5 -y

# Run compilations

COPY . /Broxy

RUN cd /Broxy \
    && git clone --depth 1 --branch v1.0.1 https://github.com/cnr-isti-vclab/vcglib.git \
    && cd vcglib \
    && git checkout v1.0.1

RUN cd /Broxy && ./compile-all.sh

ENV LD_LIBRARY_PATH=/Broxy/Bin
WORKDIR /Broxy
