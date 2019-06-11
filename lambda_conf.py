# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import mini_xception as xception
from random import shuffle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import importlib
import keras.backend as K
from keras.models import load_model
import pandas as pd
from sklearn.metrics import auc as calc_auc



%matplotlib inline


NUM_CLASSES = 7

def to_onehot(num):
    out = np.empty([0,NUM_CLASSES])
    for x in np.nditer(num):
        onehot = np.zeros(NUM_CLASSES)
        onehot[int(x)] = 1
        out = np.append(out,[onehot],axis = 0)
    return out  

def get_array_from_string(s):
    l = s.split(" ")
    l = [int(n) for n in l]
    return l
    
def get_train_test(ratio,data):
    shuffle(data)
    train = data[0:(int(len(data)*ratio[0])+1)]
    vali = data[int(len(data)*ratio[0]):int(len(data)*ratio[1])]
    test = data[int(len(data)*ratio[0]):-1]
    return train,vali,test

def remove_emo(data,num):
    new_data = []
    for item in data:
        if(item[0][0][num] == 1):
            continue
        new_data.append(item)
    return new_data

def get_metrics(acc_ar,max_pred_values,threshold=0.5):
    conf_acc_1 = (acc_ar==True)
    conf_acc_2 = (max_pred_values >= threshold)
    conf_acc_mul_tp = conf_acc_1 * conf_acc_2

    conf_acc_1 = (acc_ar==False)
    conf_acc_2 = (max_pred_values < threshold)
    conf_acc_mul_tn = conf_acc_1 * conf_acc_2

    conf_acc_1 = (acc_ar==True)
    conf_acc_2 = (max_pred_values < threshold)
    conf_acc_mul_fn = conf_acc_1 * conf_acc_2

    conf_acc_1 = (acc_ar==False)
    conf_acc_2 = (max_pred_values >= threshold)
    conf_acc_mul_fp = conf_acc_1 * conf_acc_2

    tp = conf_acc_mul_tp[conf_acc_mul_tp==True].shape[0]
    tn = conf_acc_mul_tp[conf_acc_mul_tn==True].shape[0]
    fn = conf_acc_mul_fn[conf_acc_mul_fn==True].shape[0]
    fp = conf_acc_mul_fp[conf_acc_mul_fp==True].shape[0]

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    
    return tpr,fpr

#evaluate conf prediction with max softmax
def get_max_pred_values(data_x,data_y):
    res = model.predict(x=data_x)
    pred_classes = np.argmax(res,axis=1)
    true_classes = np.argmax(data_y,axis=1)
    #acc_ar if entries are 0, accurate prediction
    acc_ar = pred_classes - true_classes
    acc_ar = (acc_ar == 0 )
    max_pred_values = np.max(res,axis=1)
    return max_pred_values,acc_ar

def get_auc_val(max_pred_values,acc_ar):
    points = []
    for i in range(0,500):
        tpr,fpr = get_metrics(acc_ar,max_pred_values,i/500)
        points.append([tpr,fpr])
    point  = np.array(points)
    auc_val = calc_auc(x=point[:,1],y=point[:,0])
    return auc_val


data = []
emo = []
pixels = []
with open('/home/sleek_eagle/research/emotion_recognition/data/fer2013/fer2013.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

extracted = []
for i,item in enumerate(data):
    if(i==0):
        continue
    emo = int(item[0])
    emo = to_onehot(emo)
    ar = np.array(get_array_from_string(item[1]))
    ar = np.reshape(ar,(48,48,1))
    extracted.append([emo,ar])

import mini_xception as xception
importlib.reload(xception)    
model = xception.get_xception()


train,vali,test = get_train_test([0.6,0.8],extracted)

train_y = np.array([np.reshape(item[0],(NUM_CLASSES,)) for item in train])
train_x = np.array([item[1] for item in train])

test_y = np.array([np.reshape(item[0],(NUM_CLASSES,)) for item in test])
test_x = np.array([item[1] for item in test])

vali_y = np.array([np.reshape(item[0],(NUM_CLASSES,)) for item in vali])
vali_x = np.array([item[1] for item in vali])

for i in range(0,1000):
    res = model.fit(x=train_x,y=train_y,epochs = 1,batch_size = 20, validation_data = [vali_x,vali_y])
    val_acc = res.history['val_acc'][0]
    if(val_acc > 0.4):
        K.set_value(model.optimizer.lr,0.0001)
        print(K.get_value(model.optimizer.lr))
    elif(val_acc > 0.5):
        K.set_value(model.optimizer.lr,0.00001)
        print(K.get_value(model.optimizer.lr))
        
    try:
        max_pred_values,acc_ar = get_max_pred_values(test_x,test_y)
        auc = get_auc_val(max_pred_values,acc_ar)

        print("auc = " + str(auc))
    except:
        pass

    
    