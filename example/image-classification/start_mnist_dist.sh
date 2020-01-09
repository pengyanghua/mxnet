#!/bin/bash
clear
./kill_train.sh
#python ../../tools/launch.py -n 1 -s 1 -H hosts --launcher ssh python train_mnist.py --kv-store dist_sync --batch-size 64 
python ../../tools/launch.py -n 1 -s 1  --launcher local python train_mnist.py --kv-store dist_sync --batch-size 64 --gpus 0 
sleep 15
cd ~/test
echo "INC_WORKER" > SCALING.txt-1

