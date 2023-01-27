# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 17:39:19 2019

@author: S.A.P
"""



import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, merge, Activation, ZeroPadding2D, concatenate
from keras.layers import AveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json
#import shutil
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib
import os                                                                                                                                    
import tensorflow as tf
from PIL import Image
#from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from plot_history import plot_history

from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix










#%%  load prediction data

# val data 
path1 = 'Dataset/img4predection'
path2 = 'Dataset/img4predection-resize'

#
#path1 = 'E:/0hatami duc/pre2019'
#path2 = 'Dataset/img4predection-resize'
#



listing = os.listdir(path1)
num_samples = np.size(listing)
print (num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((32 , 32))
    IMMGG = img.copy()
    gray = img.convert('L')
    gray.save(path2 + '\\' + file, "png")
imlistpre = os.listdir(path2)

#im1 = np.array(Image.open(path2 + '\\' + imlistVal[0]))
#m,n = im1.shape[0:2]
#imagenbr = len(imlistpre)

imMatrix_pre = np.array([np.array(Image.open(path2 + '/' + im2)).flatten() for im2 in imlistpre], 'f')

lable_pre = np.ones((num_samples,) , dtype=int)
lable_pre[0:4]=0 #Erosion
lable_pre[5:9]=1 #Polyp
lable_pre[10:14]=2 #Ulcer

pre_data, pre_lable = shuffle(imMatrix_pre[0:900] , lable_pre , random_state = 2)
predection_data = [pre_data , pre_lable]

(X_pre, y_pre) = (predection_data[0], predection_data[1])

XX_pre = X_pre.copy()

X_pre = X_pre.reshape(X_pre.shape[0], 32 , 32, 1)

X_pre = X_pre.astype('float32')


X_pre /= 255

Y_pre  = np_utils.to_categorical(y_pre, numberClass)









#%%

pre = model.predict(X_pre)

############

import numpy as np
labels_predection = np.argmax(pre, axis=1) 
labels_original = np.argmax(Y_pre, axis=1) 

###########  confusion_matrix for prediction

cm_pre=confusion_matrix(labels_predection ,labels_original)
print(cm_pre)



#%%


from sklearn.metrics import classification_report

labels_predection = np.argmax(pre, axis=1) 
labels_original = np.argmax(Y_pre, axis=1) 
target_names = ['class 0', 'class 1', 'class 2']

mesure= classification_report(labels_predection, labels_original, target_names=target_names)

print(mesure)

