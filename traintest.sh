#!/usr/bin/env bash

# Execute Training
./train.py --arch vgg --gpu True --epochs 10 --hidden_units 135 --learnrate 0.00005 --dropout 0.4 > train.log
# Execute Prediction
./predict.py --image_path flowers/valid/38/image_05829.jpg > predict.log

