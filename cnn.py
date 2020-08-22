# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:27:31 2020

@author: User
"""

import keras
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np

data = pd.read_csv('C:/Users/User/Desktop/SZABIST COURSE ASSETS/Data Science/mnist_train.csv');
print(data.head())

print(data.iloc[3,1:].values.reshape(28,28).astype('uint8')) #black and white images.
#storing pixel array in form of width and length, channel in df_x
df_x = data.iloc[:,1:].values.reshape(len(data),28,28,1)
#storing the labels in y
y = data.iloc[:,0].values

#converting labels to categories features
df_y = keras.utils.to_categorical(y,num_classes = 10)

#making categorical
df_x = np.array(df_x)
df_y = np.array(df_y)
print("making in form of matrixes")
#categorical labels
print(df_y)

print(df_x.shape)

#test train split
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y, test_size=0.2,random_state=4) 

#CNN MODEL
model = Sequential()
#convolutional layer
model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
#max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten()) 
model.add(Dense(100)) #3rd layer 200 nodes.
model.add(Dropout(0.5)) #it drop out 50% weights
model.add(Activation('softmax'))
model.add(Dense(10)) #4th layer
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

print(model.summary())
#first we train 
#fitting it with just 100 images for testing.


print(model.fit(x_train[1:100],y_train[1:100],validation_data=(x_test[1:20],y_test[1:20])))
#for better accuracy 
print(model.predict(x_test[1:20]))

print(model.evaluate(x_test,y_test))



