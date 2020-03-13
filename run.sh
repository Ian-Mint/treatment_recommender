#!/bin/bash

for j in 4 8 16 32
do
    python train.py --epochs 100 --batch_size 50 --layers 1 --lookback 4 --width $j
done

paplay /usr/share/sounds/gnome/default/alerts/glass.ogg 
