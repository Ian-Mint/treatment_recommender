#!/bin/bash

for i in 2 4 8
do
    python train.py --epochs 200 --batch 50 --layers $i --lookback 8 --width 4
done

paplay /usr/share/sounds/gnome/default/alerts/glass.ogg 
