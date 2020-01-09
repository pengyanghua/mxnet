#!/bin/bash
clear
./kill_train.sh
python ../../tools/launch.py -n 1 -s 1 -H hosts --launcher ssh python train_cifar10.py --data-train /data/mxnet-data/cifar10 --kv-store dist_sync --batch-size 64

sleep 10
cd ~/test
echo "INC_WORKER" > SCALING.txt-1


