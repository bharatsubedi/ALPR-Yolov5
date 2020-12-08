# ALPR-Yolov5 (Automatic license Plate detection and recognition)
This is the Automatic license plate detection and recognition system using Yolov5. Both plate detection and character detection and recognition using Yolov5.
I used `EnglishLP` dataset for experiment but you can try with any other dataset also

![Alt text](end-t-end-test/result/P1010002.png?raw=true "Title")
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
- Organize directories
Organize your train and val images and labels according to the example below. In this example we assume /voc is next to the /yolov5 directory. YOLOv5 locates labels automatically for each image by replacing the last instance of /images/ in the images directory with /labels/. For example:
```
voc/images/000000109622.jpg  # image
voc/labels/000000109622.txt  # label
```
For further understanding you can check inside `plate_detection` or `charater_detection`
### Select a Model
Select and modify the models parameters `nc` related to your dataset
```
plate_detection/yolov5/models/(yolov5s,yolov5m,yolov5l,yolov5x)
character_detection/yolov5/models/(yolov5s,yolov5m,yolov5l,yolov5x)
```
### Train
you can modify hyperparameter inside train.py
```
 Train YOLOv5s on COCO128 for 5 epochs
$ python train.py --img 640 --batch 16 --epochs 5 --data data/voc.yaml
```
### End to end testing
copy plate detection weight and character detection weight into `end-t-end-test/final_weight/(detection_weight).pt and end-t-end-test/recognition_model/final_weight/(character recognition weight).pt`
For more information check each folder inside 

EnglishLP dataset pretrained weight download link for 
* detection weight https://drive.google.com/file/d/1dnTTVbGq4NLDDlJHMZBAwPcRVjL_yQgZ/view?usp=sharing
* recognition weight https://drive.google.com/file/d/1aNvYucWlfVy8w_Ijh0wEE9j7rtMCKLjR/view?usp=sharing
