#!/usr/bin/env python

# Imports here
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import time
import os
from torch.optim import lr_scheduler
import copy
import datetime
import time
import json
from collections import OrderedDict
from torch.autograd import Variable


TRAIN = 'train'
VAL = 'valid'
TEST = 'test'

def getImageDataSetFn(data_dir="."):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]),
        VAL: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]),
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in [TRAIN, VAL, TEST]
    }
    return image_datasets

def getDataLoaderFn(image_datasets):

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=16,
            shuffle=True, num_workers=4
        )
        for x in [TRAIN, VAL, TEST]
    }

    return dataloaders


    # ### Label mapping
    # 

def getCatToNameFn(jsonFile):
    with open(jsonFile, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name



# TODO: Build and train your network
def build_optimizer(model,lr=0.0001):
    return optim.Adam(model.classifier.parameters(),lr=lr)
      
def build_model(activation,
                architecture_model,
                final_units,
                dropout=0.5,
                hidden_units=[512,256]):
    model = architecture_model;
    # Freeze Parameters
    for param in model.features.parameters():
        param.require_grad = False
    
    # Build network on top of pretrained model.
    length_classifier_layers = len(model.classifier)

    in_feature_count = model.classifier[0].in_features
    start_units = in_feature_count
    if len(hidden_units) > 0:
        new_features = [ nn.Linear(in_feature_count,hidden_units[0]),
                         activation,
                         nn.Dropout(p=dropout) ]
        start_units = hidden_units[0]
    else:
        new_features = []

    for i in range(len(hidden_units)):
       start_units = hidden_units[i]
       if ((i+1) < len(hidden_units)):
          end_units = hidden_units[i+1]
          new_features.extend(
              [
                  nn.Linear(start_units,end_units),
                  activation,
                  nn.Dropout(p=dropout)
              ]
          )
          start_units = end_units

       new_features.extend([nn.Linear(start_units, final_units), nn.LogSoftmax(dim=1)])
       print(new_features)
       criterion = nn.NLLLoss()
       model.classifier = nn.Sequential(*new_features)

    return (model,criterion)




def saveModelCoordinates(model,optimizer,accuracy,path="best_batch_model_coordinates.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy' : accuracy
    }, path)

def loadModelCoordinates(model,optimizer,path="best_batch_model_coordinates.pth"):
    coordinates = torch.load(path)
    model.load_state_dict(coordinates['model_state_dict'])
    optimizer.load_state_dict(coordinates['optimizer_state_dict'])
    accuracy = coordinates['accuracy']
    
    return (accuracy,model,optimizer)


def evalModel(phase,model,device,criterion,optimizer,dataloaders):
    test_loss = 0
    accuracy = 0
    model.eval()

    for ii, (images, labels) in enumerate(dataloaders[TEST]):
        images, labels = images.to(device),labels.to(device)
        logps= model.forward(images)
        loss = criterion(logps,labels)
        test_loss += loss.item()
                
        #calculate our accuracy
                
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = (top_class == labels.view(*top_class.shape))
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
    print(f"Batch: {ii+1}\t") 
    print(f"Test loss: { test_loss / len(dataloaders[phase]) :.3f}\t")
    print(f"Test percentage accuracy: {accuracy/ len(dataloaders[phase]) :.3f}")
    
    return (accuracy/ len(dataloaders[phase]))


# Train the model
def trainModel(model,device,criterion,optimizer,trainon,teston,dataloaders,epochs=5):
    current_accuracy = 0
    best_accuracy = 0
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0
        epochstart = time.time()
        for images,labels in dataloaders[trainon]:
            images, labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps,labels)
            if TRAIN == trainon:
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
        epochend = time.time()
        print('-' * 50)

        print(f"Epoch {epoch+1} took : {epochend - epochstart:.0f} seconds" )
        current_accuracy = evalModel(teston,model,device,criterion,optimizer,dataloaders=dataloaders)
        model.train()
        print(f"{datetime.datetime.now().isoformat()}\t"
            f"Epoch {epoch+1}/{epochs}\t"
            f"Train loss: {running_loss/len(dataloaders[trainon]):.4f}")
        if(best_accuracy < current_accuracy):
            best_accuracy = current_accuracy
            saveModelCoordinates(model,optimizer,best_accuracy)
            
    best_accuracy, model, optimizer =loadModelCoordinates(model,optimizer)
    
    return (best_accuracy,model,optimizer)


# Criteria for optimization negative log likelihood function for LogSoftMa

def gridSearch(device,
               criterion,
               epochs=2,
               trainon=TRAIN,
               teston=TEST,
               dropoutList=[0.1,0.2],
               activationList=[nn.ReLU()],
               layerList=[1],
               learnRateList=[0.0005,0.001]):
    # Setup optimization crteria
    best_model = None
    best_optimizer = None
    best_accuracy = 0
    for noOfLayers in layerList :
        for activation in activationList :
            for dropout in dropoutList : 
                for learnRate in learnRateList :
                    dlmodel = build_model(dropout=dropout,
                                      activation=activation,
                                      noOfLayers=noOfLayers,
                                      defaultmodel=False,
                                      append=False,
                                      modify=True)

                    optimizer = optim.Adam(dlmodel.classifier.parameters(),
                                       lr = learnRate)
                    dlmodel.to(device)
                    print("=" * 50)
                    print(f"droput={dropout}\t")
                    print(f"layers={noOfLayers}\t")
                    print(f"learnRate={learnRate}")
                    current_accuracy, current_model,current_optimizer = trainModel(dlmodel,
                                                                 device,
                                                                 criterion,
                                                                 optimizer,
                                                                 TRAIN,
                                                                 TEST,
                                                                 epochs=epochs)
                    if(best_accuracy < current_accuracy):
                        best_accuracy = current_accuracy
                        saveModelCoordinates(current_model,current_optimizer,best_accuracy, "best_grid_model_coordinates.pth")
    
    best_accuracy, dlmodel, optimizer = loadModelCoordinates(dlmodel,optimizer,"best_grid_model_coordinates.pth")
    dlmodel.train()
    dlmodel.to(device)
    return (best_accuracy, dlmodel, optimizer)



def saveCheckPoint(model,
                   optimizer,
                   epochs,dataloaders,dropout,
                   hidden_units,architecture,path):
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.epochs = epochs
    checkpoint = {'stateDictionary': model.state_dict(),
                  'outputDictionary':optimizer.state_dict(),
                  'batchSize': dataloaders['train'].batch_size,
                  'class_to_idx': model.class_to_idx,
                  'epochs': model.epochs,
                  'dropout': dropout,
                  'hidden_units' : hidden_units,
                  'architecture' : architecture
                 } 
    torch.save(checkpoint, path)

def loadCheckPoint(path):
    activation=nn.ReLU()
    checkpoint = torch.load(path)
    hidden_units= getHiddenUnits(checkpoint['hidden_units'])
    architecture = checkpoint['architecture']
    final_units=len(checkpoint['class_to_idx'])
    print(final_units)
    model,criterion = build_model(activation,
                        architecture_model=getModelFromArchitecture(architecture),
                        final_units=final_units,
                        dropout=checkpoint['dropout'],
                        hidden_units=hidden_units)
    model.load_state_dict(checkpoint['stateDictionary'])
    
    optimizer = optim.Adam(model.classifier.parameters(),
                                       lr = 0)
    optimizer.load_state_dict(checkpoint['outputDictionary'])
    model.class_to_idx = checkpoint['class_to_idx']
    print(model)

    return (model,optimizer)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    origSize= 256
    rgb=255 # from 0 -255
    
    reqSize = 224
    
    resizedimage = getResizedImage(image,resize=origSize)
    
    procimage = resizedimage.crop((int((origSize - reqSize)/2),
                        int((origSize - reqSize)/2),
                        int((origSize + reqSize)/2),
                        int((origSize + reqSize)/2)))
    numpyImage = np.array(procimage)
    numpyImage = numpyImage/float(rgb)
        
    image1 = numpyImage[:,:,0]
    image2 = numpyImage[:,:,1]
    image3 = numpyImage[:,:,2]
    
    #Normalization
    normalizedImage1 = (image1 - 0.485)/(0.229) 
    normalizedImage2 = (image2 - 0.456)/(0.224)
    normalizedImage3 = (image3 - 0.406)/(0.225)
        
    numpyImage[:,:,0] = normalizedImage1
    numpyImage[:,:,1] = normalizedImage2
    numpyImage[:,:,2] = normalizedImage3
    
    numpyImage = np.transpose(numpyImage, (2,0,1))
    
    return numpyImage


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(model, pred_image_path, device,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print(device)
    if device == 'cuda':
        model.to(device)
    image = torch.FloatTensor([process_image(Image.open(pred_image_path))])
    model.eval()
    if device == 'cuda':
       variableimage = Variable(image).cuda()
    else:
       variableimage = Variable(image)

    output = model.forward(variableimage)
    if device == 'cuda':
        pobabillityList = torch.exp(output).cpu().data.numpy()[0]
    else:
        pobabillityList = torch.exp(output).data.numpy()[0]
    

    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    top_idx = np.argsort(pobabillityList)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probablillity = pobabillityList[top_idx]

    return top_probablillity, top_class
    # TODO: Implement the code to predict the class from an image file


def view_classify(imagefile, probabillityList, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img_filename = imagefile.split('/')[-2]
    imageobj = Image.open(imagefile)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    flower_name = mapper[img_filename]
    
    ax1.set_title(flower_name)
    ax1.imshow(imageobj)
    ax1.axis('off')
    
    y_pos = np.arange(len(probabillityList))
    ax2.barh(y_pos, probabillityList)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in classes])
    ax2.invert_yaxis()

def getModelFromArchitecture(arch):
    if arch.lower() == 'vgg':
        return models.vgg16(pretrained=True)
    else:
        return None

def getHiddenUnits(strUnitDelimited):
    new_hidden_units=[]
    for item in strUnitDelimited.split(","):
        new_hidden_units.append(int(item))
    return new_hidden_units

def getResizedImage(image,resize=256):
    currHeight, currWidth = image.size
    print("In Image Size: " + str(image.size))
    new_image = None
    if currWidth < currHeight:
        newHeight = int(currHeight * resize / currWidth)
        print("newHeight :" + str(newHeight))
        new_image = image.resize((resize,newHeight))
    else:
        newWidth = int(currWidth * resize / currHeight)
        print("newWidth: " + newWidth)
        new_image = image.resize((newWidth,resize))
                     
    print("Out Image Size: " + str(new_image.size))
    return new_image
