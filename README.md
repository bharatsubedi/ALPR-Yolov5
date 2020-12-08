# ALPR-Yolov5 (Plate detection and character recognition)
This is the Automatic license plate detection and recognition system using Yolov5. Both plate detection and character detection and recognition using Yolov5.

# dependencies
```
NVIDIA GPUs
NVIDIA graphic driver
CUDA toolkit
cuDNN library
OpenCV 
Cython
matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
pillow
PyYAML>=5.3
scipy>=1.4.1
tensorboard>=2.2
torch>=1.6.0
torchvision>=0.7.0
tqdm>=4.41.0
```

This project is tested under following conditions:
```
Ubuntu 20.04 LTS
NVIDIA GeForce RTX 2080 Super
NVIDIA graphic driver version: 440.95.01
CUDA version: 10.2
cuDNN version: 7.6.5
OpenCV version: 4.2
```
## Installations

### Install NVIDIA graphic driver, CUDA toolkit, cuBLAS, and cuDNN

### Configure conda virtual environment

## Plate detection training for custom custom data

This guide explains how to train your own custom dataset with YOLOv5 for plate detection

## Before You Start
Clone this repo, prepared datasetin yolo format( you can check inside voc folder for sample image and label), and install dependencies, including Python>=3.8 and PyTorch>=1.7.
```
$ git clone https://github.com/bharatsubedi/ALPR-Yolov5  # clone repo
```
# Train On Custom Data
- update `data/voc.yaml`
   `data/voc.yaml`, shown below, is the dataset configuration file that defines 1) a path to a directory of training images (or path to a *.txt file with a list of training images), 2) the same for our validation images, 3) the number of classes, 4) a list of class names:
```
train: voc/images/
val: voc/images/

# number of classes
nc: 1

# class names
names: ['plate']
```
- create labels: 
   After using a tool labelImg to label your images, export your labels to YOLO format, with one *.txt file per image (if no objects in image, no *.txt file is required). The *.txt file specifications are:

    * One row per object
    * Each row is `class x_center y_center width height` format.
    * Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.
    * Class numbers are zero-indexed (start from 0).
