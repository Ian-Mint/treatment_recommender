#!/bin/bash

for j in 4 8 32
do
    for i in 4 8 16
    do
	python train.py --epochs 100 --batch_size 50 --layers 2 --lookback $i --width $j
    done
done

paplay /usr/share/sounds/gnome/default/alerts/glass.ogg 
