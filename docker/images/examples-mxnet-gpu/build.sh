# build and push
docker build -t yrchen/k8s-mxnet-gpu-measurement:latest -f k8s-mxnet-gpu-measurement.Dockerfile .
docker push yrchen/k8s-mxnet-gpu-measurement:latest
