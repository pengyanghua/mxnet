# build a distributed mxnet-gpu image

FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04

MAINTAINER yhpeng

RUN apt-get update
RUN apt-get -y remove python-pip python-numpy

RUN apt-get install -y htop iotop iftop nload iperf curl dnsutils wget
RUN apt-get install -y build-essential git libatlas-base-dev libopencv-dev python-opencv libcurl4-openssl-dev libgtest-dev cmake wget unzip
RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib

RUN apt-get install -y libopenblas-dev liblapack-dev libopencv-dev

RUN apt-get install -y python-dev python-setuptools

RUN wget https://bootstrap.pypa.io/get-pip.py && python2 get-pip.py && rm get-pip.py
RUN pip2 install numpy

RUN mkdir /root/.ssh/
ADD id_rsa /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts
RUN chmod 700 /root/.ssh/id_rsa
RUN printf "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config

ENV BUILD_OPTS "USE_PROFILER=1 USE_OPENCV=1 USE_BLAS=openblas USE_DIST_KVSTORE=1 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_MKL2017=1 USE_MKL2017_EXPERIMENTAL=1 MKLML_ROOT=/mxnet/mkl"
ENV LD_LIBRARY_PATH /mxnet/mkl/lib:$LD_LIBRARY_PATH
RUN git clone --recursive git@github.com:yhpeng-git/elastic-mxnet.git
RUN mv elastic-mxnet mxnet && cd mxnet && make -j $(nproc) $BUILD_OPTS && \
    cd python && python2 setup.py build && python2 setup.py install


