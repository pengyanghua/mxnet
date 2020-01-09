#!/bin/bash
clear
./kill_train.sh
echo "" > ~/test/overhead.txt
python ../../tools/launch.py -n 2 -s 3 -H hosts --launcher ssh python train_imagenet.py --data-train /data/mxnet-data/imagenet --kv-store dist_sync --batch-size 64  --gpus 0 &

sleep 30
cd ~/test
#echo "INC_SERVER" > SCALING.txt-1
#sleep 5 
#echo "NONE" > SCALING.txt-1
#sleep 100
#echo "INC_SERVER" > SCALING.txt-1

