echo "compile mxnet GPU version..."
#make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_DIST_KVSTORE=1 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda-8.0
make -j $(nproc) USE_PROFILER=1 USE_OPENCV=1 USE_BLAS=openblas USE_DIST_KVSTORE=1 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda-8.0 
 cd python && sudo python setup.py install

