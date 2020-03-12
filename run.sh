#!/bin/bash

for i in 16 32
do
    python train.py --epochs 200 --batch 50 --layers 1 --lookback $i
done
