docker login
docker build -t yhpeng/elastic-mxnet-gpu:latest -f elastic-mxnet-gpu.Dockerfile .
docker push yhpeng/elastic-mxnet-gpu:latest
