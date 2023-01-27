# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 17:35:40 2019

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
from sklearn.metrics import confusion_matrix#%%
# mini batch size 
bath_Size =32

numberClass = 3

NumberEpoch =200

img_Channels = 3

#%%

# load data train





path1 = 'Dataset/train'
path2 = 'Dataset/trainResized'

listing = os.listdir(path1)
num_samples = np.size(listing)
print (num_samples,'num_samples')

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((32 , 32))
    IMMGG = img.copy()
    gray = img.convert('L')
    gray.save(path2 + '\\' + file, "png")
imlist = os.listdir(path2)

im1 = np.array(Image.open(path2 + '\\' + imlist[0]))
m,n = im1.shape[0:2]
imagenbr = len(imlist)


imMatrix = np.array([np.array(Image.open(path2 + '\\' + im2)).flatten() for im2 in imlist], 'f')

lable_train = np.ones((num_samples,) , dtype=int)
lable_train[0:908]=0 #Erosion
lable_train[909:1824]=1 #Polyp
lable_train[1825:2766]=2 #Ulcer

data, lable = shuffle(imMatrix , lable_train , random_state = 2)
train_data = [data , lable]


#%%
# The data, split between train and test sets:


datanum=len(data)
test_num=round(datanum*0.25)
train_num= (datanum-test_num)


X_test = data[0:test_num]
Y_test = lable[0:test_num]

X_train = data[test_num:]
Y_train = lable[test_num:]


print('X_train shape:', X_train.shape)
print(train_num, 'train samples')
print('X_test shape:', X_test.shape)
print(test_num, 'test samples')



X_train = X_train.reshape(X_train.shape[0], 32 , 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32 , 32, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, numberClass)
Y_test  = np_utils.to_categorical(Y_test, numberClass)



#%%  load validation data

# val data 
path1 = 'Dataset/validation-new'
path2 = 'Dataset/validation_resize'

listing = os.listdir(path1)
num_samples = np.size(listing)
print (num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((32 , 32))
    IMMGG = img.copy()
    gray = img.convert('L')
    gray.save(path2 + '\\' + file, "png")
imlistVal = os.listdir(path2)

im1 = np.array(Image.open(path2 + '\\' + imlistVal[0]))
m,n = im1.shape[0:2]
imagenbr = len(imlistVal)

imMatrixVal = np.array([np.array(Image.open(path2 + '/' + im2)).flatten() for im2 in imlistVal], 'f')

lable_val = np.ones((num_samples,) , dtype=int)
lable_val[0:299]=0 #Erosion
lable_val[300:599]=1 #Polyp
lable_val[600:899]=2 #Ulcer

val_data, val_lable = shuffle(imMatrixVal[0:900] , lable_val , random_state = 2)
validation_data = [val_data , val_lable]

(X_val, y_val) = (validation_data[0], validation_data[1])

XX_test = X_val.copy()

X_val = X_val.reshape(X_val.shape[0], 32 , 32, 1)

X_val = X_val.astype('float32')


X_val /= 255

Y_val  = np_utils.to_categorical(y_val, numberClass)

#%%
def FireModule(s_1x1, e_1x1, e_3x3, name):
    """
        Fire module for the SqueezeNet model. 
        Implements the expand layer, which has a mix of 1x1 and 3x3 filters, 
        by using two conv layers concatenated in the channel dimension. 
        Returns a callable function
    """
    def layer(x):
        squeeze = keras.layers.Convolution2D(s_1x1, 1, 1, activation='relu', init='he_normal', name=name+'_squeeze')(x)
        squeeze = keras.layers.BatchNormalization(name=name+'_squeeze_bn')(squeeze)
        # Set border_mode to same to pad output of expand_3x3 with zeros.
        # Needed to merge layers expand_1x1 and expand_3x3.
        expand_1x1 = keras.layers.Convolution2D(e_1x1, 1, 1, border_mode='same', activation='relu', init='he_normal', name=name+'_expand_1x1')(squeeze)
        # expand_1x1 = BatchNormalization(name=name+'_expand_1x1_bn')(expand_1x1)

        # expand_3x3 = ZeroPadding2D(padding=(1, 1), name=name+'_expand_3x3_padded')(squeeze)
        expand_3x3 = keras.layers.Convolution2D(e_3x3, 3, 3, border_mode='same', activation='relu', init='he_normal', name=name+'_expand_3x3')(squeeze)
        # expand_3x3 = BatchNormalization(name=name+'_expand_3x3_bn')(expand_3x3)

        expand_merge = keras.layers.concatenate([expand_1x1, expand_3x3], axis=3, name=name+'_expand_merge')
        return expand_merge
    return layer

#%% layers

input_image = keras.layers.Input(shape=(32,32,1))

padd_conv1 = keras.layers.ZeroPadding2D(padding=(2, 2))(input_image)

conv1 =keras.layers.Conv2D(32, (3, 3), activation='relu',strides=(1, 1),)(padd_conv1)

maxpool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

###########
fire2 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire2')(maxpool1)
fire3 = FireModule(s_1x1=16, e_1x1=32, e_3x3=32, name='fire3')(fire2)

########


padd_conv4 = keras.layers.ZeroPadding2D(padding=(1, 1))(fire3)

conv4 = keras.layers.Conv2D(3, (1, 1), activation='relu', subsample=(2, 2), init='he_normal', name='conv2' )(padd_conv4)

Averagepool4 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3')(conv4)


out_dropout = Flatten()(Averagepool4)
out1=keras.layers.Dense(3)(out_dropout)

softmax = keras.layers.Activation('softmax', name='softmax')(out1)
model = Model(input=input_image, output=[softmax])


#%%

# Tiiiiime !!


import datetime
start = datetime.datetime.now()





# Compile model

#  Stochastic gradient descent optimizer
sgd = optimizers.SGD(lr=0.001, decay=1e-9, momentum=0.9)

#A metric is a function that is used to judge the performance of your mode


model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
#history = model.fit(X_train, Y_train , validation_split=0.33, bath_Size, NumberEpoch)


#By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.

history = model.fit(X_train, Y_train, bath_Size, NumberEpoch, verbose=1, validation_data=(X_val,Y_val))


#y_prediction = model.fit(X_train, y_train).predict(X_test)



#
plot_history(history)


#history = model.fit(X_train, Y_train, bath_Size, NumberEpoch, verbose=1, validation_split=0)

#record the training / validation loss / accuracy at each epoch
loss_acc=history.history
#print(history.history)
loss = history.history['loss']
print (loss)
acc = history.history['acc']
print(acc)



model.summary()

model.save('my_Trained_model.h5')
print('Ok, Model Saved!!')


###  END Time
end = datetime.datetime.now()
traintime = end - start
print('Total training time: ', str(traintime))



#%%
##   plot
#history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()





