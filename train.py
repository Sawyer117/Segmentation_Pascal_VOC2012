# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:36:33 2020

@author: 36284
"""

# USAGE
# python train.py --dataset Sports-Type-Classifier/data --model model/activity.model --label-bin model/lb.pickle --epochs 50

# set the matplotlib backend so figures can be saved in the background
#import matplotlib
#matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
#import argparse
import pickle
import cv2
#import pandas 
import random

from model import unet
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']="2" #specify which GPU(S) to be used
#%% Setting Parameters
illustate_result = False
num_epochs = 1
model_path = './model/Res50_Violence.h5'
label_bin = './model/lb.pickle'
image_size = 224
Batch_Size = 32
# construct the argument parser and parse the arguments
# =============================================================================
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output serialized model")
# ap.add_argument("-l", "--label-bin", required=True,
# 	help="path to output label binarizer")
# ap.add_argument("-e", "--epochs", type=int, default=25,
# 	help="# of epochs to train our network for")
# ap.add_argument("-p", "--plot", type=str, default="plot.png",
# 	help="path to output loss/accuracy plot")
# args = vars(ap.parse_args())
# =============================================================================
# %%  Read Pascal VOC 2012 dataset
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


#%%
from sklearn.utils import shuffle
def make_image_gen(list_imgs,image_size,batch_size):
    SEED = np.random.seed(np.random.choice(range(9999)))
    random.seed(SEED)
    random.shuffle(list_imgs)
    out_rgb,out_label = [],[]
    while True:
        #seq_X, seq_Y = shuffle(X, Y, random_state=SEED)
        for idx, data in enumerate(list_imgs):
            train_img_path = os.path.join(PASCAL_DIR,'VOC2012/JPEGImages/',str(data.strip()+".jpg"))
            gt_path = os.path.join(PASCAL_DIR,"VOC2012/SegmentationClass/",str(data.strip()+".png"))
            image = cv2.imread(train_img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_size, image_size))
            gt_image = cv2.imread(gt_path,0)
            gt_image = cv2.resize(gt_image, (image_size, image_size))
            gt_image = gt_image[..., np.newaxis]
            out_rgb += [image]
            out_label += [gt_image]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb,0)/255.0, np.stack(out_label,0)/255.0
                out_rgb, out_label = [], []
    
train_data_generator = make_image_gen(train_set, 224, 8)
valid_data_generator = make_image_gen(val_set, 224, 8)
#train_x, train_y = next(train_data)


#%% 
# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
#mean = np.array([123.68, 116.779, 103.939], dtype="float32")
#trainAug.mean = mean
#valAug.mean = mean

def aug_gen(in_gen):
    for in_x, in_y in in_gen:
        #g_x = trainAug.flow(255*in_x,batch_size = Batch_Size)
        g_x = in_x
        g_y = in_y
        
        yield next(g_x), g_y
        
#cur_train_gen = aug_gen(train_data_generator)

#t_x, t_y = next(cur_gen)









#%% Load keras model here
model = unet(None,(224,224,3))


#%%
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")
H = model.fit_generator(
	train_data_generator,
	steps_per_epoch=len(train_set) // Batch_Size,
	validation_data=valid_data_generator,
	validation_steps=len(val_set) // Batch_Size,
	epochs=num_epochs)



# serialize the model to disk
print("[INFO] serializing network...")
model.save(model_path)

# serialize the label binarizer to disk
f = open(label_bin, "wb")
f.write(pickle.dumps(lb))
f.close()

# evaluate the network
# =============================================================================
print("[INFO] evaluating network...")
a, b = next(make_image_gen(testX, testY, 224, 50))
predictions = model.predict(a, batch_size=50)
report = classification_report(b,
	predictions, target_names=lb.classes_, output_dict = True)
print(report)
df = pandas.DataFrame(report).transpose()
df.to_csv('report.csv')
#print(classification_report(testY.argmax(axis=1),
    #predictions.argmax(axis=1), target_names=lb.classes_))
# =============================================================================







# plot the training loss and accuracy
if illustate_result == True:
    N = num_epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    #plt.savefig(args["plot"])