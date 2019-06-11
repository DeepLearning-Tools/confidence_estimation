#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:12:36 2019

@author: sleek_eagle
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalAveragePooling3D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import Add
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers import Lambda
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.merge import concatenate
from keras import layers
from keras.regularizers import l2
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.losses import mean_squared_error
from keras.metrics import categorical_accuracy
from keras import optimizers

import numpy as np

import tensorflow as tf

K.clear_session()




# parameters
input_shape = (48, 48, 1)
label_shape = (7,1)
verbose = 1
num_classes = 7
patience = 50
base_path = 'models/'
l2_regularization=0.01



# model parameters
regularization = l2(l2_regularization)


def get_xception_conf():
    #take grouargnd truth labels as input for calculating the loss fucntion
    label_input = Input(label_shape,name='label_input')
    #take lambda as input to the network
    lmd_input = Input((1,1),name='lambda_input') 
    
    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    pred = Activation('softmax',name='predictions')(x)
    
    #confidence network

    conf_dense_1    = layers.Dense(name = 'conf_dense_1', units = 20, use_bias = True,activation = 'relu')(x)
    conf_dense_2    = layers.Dense(name = 'conf_dense_2', units = 5, use_bias = True,activation = 'relu')(conf_dense_1)
    confidence = layers.Dense(1, activation='sigmoid',name = 'confidence')(conf_dense_2)
    confidence = Reshape((1,))(confidence)

    
    #calculate losses
    parameters = layers.Lambda(calc_param)([label_input,pred,confidence,lmd_input])
    loss_L = layers.Lambda(get_L)(parameters)
   
    #merge two networks together
    single_out = concatenate([pred,confidence])
    
    model = Model(inputs = [img_input,label_input,lmd_input], outputs = [single_out,parameters])

    sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(optimizer=sgd, loss=loss_func(loss_L),loss_weights = [1,0],metrics=[accuracy_without_conf])

    return model

def get_xception():
    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    sgd = optimizers.SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=False)

    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def calc_param(inputs):
    num_classes = 7
    label_input = inputs[0]
    label_input = K.squeeze(label_input,axis=2)

    pred = inputs[1]
    confidence = inputs[2]
    lmd = inputs[3][0]
    lmd =  K.squeeze(lmd,axis=1)

    #K.log is ln i.e log to the base of 2
    L_c = -K.log(confidence)
    L_c =  K.squeeze(L_c,axis=1)
    #interpolated value with c
    confPred_mat = K.tf.tile(confidence,[1,num_classes])

    p_interp = K.tf.multiply(confPred_mat,pred) + K.tf.multiply((1-confPred_mat),label_input)
    
    L_t = K.categorical_crossentropy(label_input,p_interp)
    L = L_t + lmd* L_c
    a = K.variable([1])
    b = K.variable([2])
    c = K.variable([3])
    d = K.variable([4])

    L = K.expand_dims(L,axis=1)
    L_t = K.expand_dims(L_t,axis=1)
    L_c = K.expand_dims(L_c,axis=1)

    conc = K.concatenate(tensors = [L,L_t,L_c],axis=1)
    
    return conc

def get_L(inputs):
    return inputs[:,0]  

def loss_func(L):
    def loss(yTrue,Ypred):
        nonlocal L
        return L
    return loss

def get_zero_loss(yTrue,yPred):
    return K.zeros(shape=[])

def accuracy_without_conf(yTrue,yPred):
    yT = yTrue[:,0:num_classes]
    yP = yPred[:,0:num_classes]
    return categorical_accuracy(yT,yP)
        

