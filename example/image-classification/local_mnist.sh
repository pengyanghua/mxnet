#!/bin/bash
EXECFILE=`pwd`
EXECFILE+='/train_mnist.py'
python $EXECFILE  --kv-store local --batch-size 64
