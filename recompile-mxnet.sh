
nvidia-smi
if [ $? -eq 0 ]; then
    echo "compile mxnet GPU version..."
    make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_DIST_KVSTORE=1 
else
    echo "compile mxnet CPU version..."
    make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_DIST_KVSTORE=1 USE_MKL2017=1 USE_MKL2017_EXPERIMENTAL=1 MKLML_ROOT=$MKLML_ROOT
fi

echo "compile python binding..."
cd python
python setup.py build
sudo python setup.py install
