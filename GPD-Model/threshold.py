# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:33:17 2019

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
from keras.models import model_from_json
from keras.models import load_model
#%%

del model


#weights=model.load_weights('weights_asli.h5')

model=load_model('model_asli.h5')
weights= model.get_weights()

weight_org=model.get_weights()

sparsified_weights = []
th =  0.001
#####################

for w in weights:
   aa = (abs(w) > th)
   sparsified_weights.append(w*aa)

model.set_weights(sparsified_weights)

weights= model.get_weights()

 
   
#%%
#############################################################################

#%%
# mini batch size  
bath_Size =32

numberClass = 3

NumberEpoch =50


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





#%%%


import datetime
start = datetime.datetime.now()
sgd = optimizers.SGD(lr=0.001, decay=1e-9, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
#for m in range (0,10):
#    history = model.fit(X_train, Y_train, bath_Size, NumberEpoch, verbose=1, validation_data=(X_val,Y_val))

##### Step  DECAY TEEEEST
from keras.callbacks import LearningRateScheduler
import math

def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop =50
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))


loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate]


##### END

#By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
#for m in range (0,6):
history = model.fit(X_train, Y_train, bath_Size, NumberEpoch,callbacks=callbacks_list, verbose=1, validation_data=(X_val,Y_val))


 
#record the training / validation loss / accuracy at each epoch
loss_acc=history.history
#print(history.history)
loss = history.history['loss']
#print (loss)
acc = history.history['acc']
#print(acc)





###  END Time
end = datetime.datetime.now()
traintime = end - start
print('Total training time: ', str(traintime))







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

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(X_test, Y_test, batch_size=32)
print('test loss, test acc:', results)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on val data')
results = model.evaluate(X_val, Y_val, batch_size=32)
print('test loss, test acc:', results)


#######

model.save_weights('weights_th.h5')
model.save('model_th.h5')






#%%    تسسسسسسسسسسسسسسسسست

#%%




# Check the trainable status of the individual layers
#for layer in model.layers:
#    print(layer, layer.trainable)
#
#
#
#for layer in model.layers:
#    weights = layer.get_weights()
#
#
#
#
#
#from keras.utils import plot_model
#plot_model(model, to_file='/model.png', show_shapes=True,)







###################### OOOOOOOKKKKKKKKKKK
#for list in weights:
#    for number in list:
#            print (number)
#
#
### for check
#
#temp = []
#for list in weights:
#    for number in list:
#            temp.append(number)       
#            documents = temp
#

        
        
#        ############### :) 
#        
#for c in range (0,1):          
#    for z in range (0,1):     
#        for a in range (0,64):
#            for b in range (0,3):
#                if (np.array(weights[24][c][z][a][b]) > 0.001):
#                    weights[24][c][z][a][b] = 0
#                    print(weights[24][c][z][a][b])
#                else:
#                    print("nothing")
#    
#
#for c in range (0,len(weights)):
#    for z in range (0,len(weights[c])):   
#        if (isinstance(weights[c][z], list)):
#            for a in range (0,len(weights[c][z])):
#                if (isinstance(weights[c][z][a], list)):
#                    for b in range (0,len(weights[c][z][a])):
#                        if (isinstance(weights[c][z][a][b], list)):
#                            for q in range (0,len(weights[c][z][a][b])):
#                                if ((weights[c][z][a][b][q]) < 0):
#                                    print(c,z,a,b,q) 
#                                    weights[c][z][a][b][q] = 0
#                                    print(weights[c][z][a][b][q])
#        else :
#            if ((weights[c][z]) < 0):
#                weights[c][z] = 0
#                print(c,z) 
#                
#for a in range (13,18):    
#    for b in range (0,3):
#                    if (np.array(weights[25][b]) > 0.001):
#                        weights[25][b] = 0
#                        print(weights[25][b])    
#        

   ################################################ 
    



