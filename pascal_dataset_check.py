# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:30:28 2020

@author: 36284
"""

import os
import glob
from random import sample
import matplotlib.pyplot as plt
import cv2

#path to pascal_voc 2012, point to VOCdevkit folder please
PASCAL_DIR = "E:/Coding Cache/pascal_voc_segmentation/data/VOCdevkit/"

train_txt = os.path.join(PASCAL_DIR,"VOC2012/ImageSets/Segmentation/train.txt")
file = open(train_txt,"r")
train_set = file.readlines()
file.close()
print("Found %d items for trainning"%len(train_set))

val_txt = os.path.join(PASCAL_DIR,"VOC2012/ImageSets/Segmentation/val.txt")
file = open(val_txt,"r")
val_set = file.readlines()
file.close()
print("Found %d items for validation"%len(val_set))

#Show 5 random training img/gt label pair
NUMBER_SAMPLE=5
show_case = sample(train_set,NUMBER_SAMPLE)
fig=plt.figure(figsize=(20, 20))
for idx,item in enumerate(show_case):
    train_path = os.path.join(PASCAL_DIR,'VOC2012/JPEGImages/',str(item.strip()+".jpg"))
    gt_path = os.path.join(PASCAL_DIR,"VOC2012/SegmentationClass/",str(item.strip()+".png"))
    img = cv2.imread(train_path)
    gt = cv2.imread(gt_path,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    print("train img height,width: %d,%d"%(img.shape[0],img.shape[1]))
    print("val img height,width: %d,%d"%(gt.shape[0],gt.shape[1]))
    fig.add_subplot(NUMBER_SAMPLE, 2, 2*idx+1)
    plt.imshow(img)
    fig.add_subplot(NUMBER_SAMPLE, 2, 2*idx+2)
    plt.imshow(gt)
    print(idx)
plt.show()
