#!/usr/bin/env python
from ImageClassifierUtils import predict
from ImageClassifierUtils import loadCheckPoint
from ImageClassifierUtils import getCatToNameFn

import argparse

def getDevice(gpu):
    if gpu:
        return "cuda"
    else:
        return "cpu"


def main():
    
    parser = argparse.ArgumentParser(description='Train Image Classfier')
    parser.add_argument('--mapper',type=str,default='cat_to_name.json',help='mapper file')
    parser.add_argument('--checkpoint_dir' , type=str, default='final_model_checkpoint_cmd.pth', help='model checkpoint directory')
    parser.add_argument('--image_path', type=str, help='path of image')
    parser.add_argument('--topk', type=int, default=10, help='calculate top k probabilities')
    parser.add_argument('--gpu', type=bool, default=False, help='Is GPU Enabled')


    args = parser.parse_args()
    print(args)

    #------ load checkpoint --------#
    model, optimizer = loadCheckPoint(path = args.checkpoint_dir)

    top_probability, top_class = predict(model=model,pred_image_path=args.image_path,topk=args.topk,device=getDevice(args.gpu))

    print('Predicted Classes: ', top_class)
    print ('Class Names: ')
    cat_to_name = getCatToNameFn(args.mapper)
    [print(cat_to_name[x]) for x in top_class]
    print('Predicted Probability: ', top_probability)


if __name__ == "__main__":
   main()
