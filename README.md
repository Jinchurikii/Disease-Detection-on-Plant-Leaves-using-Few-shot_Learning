# Disease-Detection-on-Plant-Leaves-using-Few-shot_Learning


This repository contains the code and resources for my master's thesis, which focuses on few-shot object detection to identify unhealthy leaves in the agriculture domain.

## Setup the environment


- Clone github repo for ultralytics yolov5 https://github.com/ultralytics/yolov5
- Install all the packages in requirement.txt
 
 
## Implimentation

- For implementation of Yolov5, please follow the instructions of ultralytics yolov5 repo. I have changed some hyperparameters according to      my training requirements and dataset. 
- For YOLOv5, train.py is used to train on DFKI GPU clusters. 
- For testing, a separate test.py script provided by yolov5 model can be used.

## Pre-trained weights

- The pre-trained weights for YOLOv5 trained on PlantVillage dataset from scratch, and with coco weights are available in the weight folder.
- Those weights can be used for fine-tuning the model with any other crop dataset or few-shot crop dataset.



## Acknowledgments

- The PlantVillage dataset was used for this study.
- The YOLOv5 repository provided the baseline model and training code.
















