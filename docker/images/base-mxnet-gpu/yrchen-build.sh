docker login
docker build -t chenyr13/elastic-mxnet-gpu:latest -f elastic-mxnet-gpu.Dockerfile .
docker push chenyr13/elastic-mxnet-gpu:latest
