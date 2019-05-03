#!/usr/bin/env python

#==============|==============================================================#
# Name:        | train.py                                                     #
#              |                                                              #
# Author:      | Vijaykumar Singh                                             #
#              |                                                              #
# Description: | The module trains a neural network using transfer learning   #
#              | for classifying flower images to correct class labels.       #
#==============|==============================================================#

from ImageClassifierUtils import build_model
from ImageClassifierUtils import build_optimizer
from ImageClassifierUtils import evalModel
from ImageClassifierUtils import getCatToNameFn
from ImageClassifierUtils import getDataLoaderFn
from ImageClassifierUtils import getImageDataSetFn
from ImageClassifierUtils import getModelFromArchitecture
from ImageClassifierUtils import loadCheckPoint
from ImageClassifierUtils import saveCheckPoint
from ImageClassifierUtils import trainModel
from ImageClassifierUtils import getHiddenUnits
import argparse
from torch import nn


def getDevice(gpu):
    if gpu:
        return "cuda"
    else:
        return "cpu" 

def main():
    parser = argparse.ArgumentParser(description='Train Image Classfier')
    parser.add_argument('--gpu', type=bool, default=False, help='Is GPU Enabled')
    parser.add_argument('--arch', type=str, default='vgg', help='Model architecture [available: vgg,densenet]', required=True)
    parser.add_argument('--learnrate', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--hidden_units', type=str, default='512,256', help='hidden units in layers')
    parser.add_argument('--dropout',type=float,default=0.25,help="Dropout for network")
    parser.add_argument('--epochs', type=int, default=7, help='training Epochs')
    parser.add_argument('--data_dir', type=str, default='flowers', help='dataset path')
    parser.add_argument('--mapper',type=str,default='cat_to_name.json',help='mapper file')
    parser.add_argument('--checkpoint_dir' , type=str, default='final_model_checkpoint_cmd.pth', help='model checkpoint directory')
    args = parser.parse_args()
    print(args)

    #Initialize Train,Test and Val
    TRAIN = 'train'
    VAL = 'valid'
    TEST = 'test'


    # These defaults are assumed within program
    activation = nn.ReLU()

    #Loaddatasets to train models.
    image_dataset = getImageDataSetFn(args.data_dir)

    #LoadDatasets to DataLoaders
    dataloaders= getDataLoaderFn(image_dataset)

    cat_to_name = getCatToNameFn(args.mapper)
    #print(cat_to_name)
    final_units = len(cat_to_name)
    hidden_units= getHiddenUnits(args.hidden_units)
    model,criterion = build_model(activation,
                        architecture_model=getModelFromArchitecture(args.arch),
                        final_units=final_units,
                        dropout=args.dropout,
                        hidden_units=hidden_units)
    # Now that built model is available, let it train
    optimizer = build_optimizer(model,lr=args.learnrate)
    trainedAccuracy, trainedModel, trainedOptimizer= trainModel(model,
                                                          getDevice(args.gpu),
                                                          criterion,
                                                          optimizer,
                                                          TRAIN,
                                                          TEST,
                                                          epochs=args.epochs,
                                                          dataloaders=dataloaders)

    # Once the training has been completed, perform evaluation on TEST and VAL datasets
    print("Test Results are as follows:")
    evalModel(TEST,trainedModel,getDevice(args.gpu),criterion,trainedOptimizer,dataloaders=dataloaders)
    print("Validations Results are as follows:")
    evalModel(VAL,trainedModel,getDevice(args.gpu),criterion,trainedOptimizer,dataloaders=dataloaders)


    saveCheckPoint(model=trainedModel,
                   optimizer=trainedOptimizer,
                   epochs=args.epochs,
                   dropout=args.dropout,
                   dataloaders=dataloaders,
                   path=args.checkpoint_dir,
                   architecture=args.arch,
                   hidden_units=args.hidden_units)


if __name__ == "__main__":
    main()
