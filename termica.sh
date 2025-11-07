#!/bin/sh


export VIDEO=data/rooms/termica
yoloplay --video $VIDEO.mkv --save $VIDEO.csv
yolotrain --csv $VIDEO.csv --model-path $VIDEO.pkl --grid-search

echo yoloplay --video $VIDEO --svm-model data/rooms/termica.pkl
