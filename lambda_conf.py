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
    return auc_val,points

def get_per_class_acc(pred_classes):
    pred_classes = np.reshape(pred_classes,(pred_classes.shape[0],1))
    acc_ar = np.reshape(acc_ar,(acc_ar.shape[0],1))
    res = np.append(pred_classes,values = acc_ar,axis=1)
    res = pd.DataFrame(res)
    res.groupby([0]).mean()
    return res

def plot_ROC(points):
    points = np.array(points)
    plt.scatter(x=points[:,1],y=points[:,0])
    plt.ylabel("True positive rate")
    plt.xlabel("False positive Rate")
    plt.show()
    
#****************************************
#**** data preparation and training for max-confidence prediction method (vanila)
#****************************************
def get_data_vanila():
    train,vali,test = get_train_test([0.6,0.8],extracted)
    
    train_y = np.array([np.reshape(item[0],(NUM_CLASSES,)) for item in train])
    train_x = np.array([item[1] for item in train])
    
    test_y = np.array([np.reshape(item[0],(NUM_CLASSES,)) for item in test])
    test_x = np.array([item[1] for item in test])
    
    vali_y = np.array([np.reshape(item[0],(NUM_CLASSES,)) for item in vali])
    vali_x = np.array([item[1] for item in vali])
    
    return train_x,train_y,vali_x,vali_y,test_x,test_y

def train_vanila():
    train_x,train_y,vali_x,vali_y,test_x,test_y = get_data_vanila()
    for i in range(0,1000):
        res = model.fit(x=train_x,y=train_y,epochs = 1,batch_size = 20, validation_data = [vali_x,vali_y])
        val_acc = res.history['val_acc'][0]
        if(val_acc > 0.4):
            K.set_value(model.optimizer.lr,0.001)
            print(K.get_value(model.optimizer.lr))
        elif(val_acc > 0.5):
            K.set_value(model.optimizer.lr,0.0001)
            print(K.get_value(model.optimizer.lr))
            
        try:
            max_pred_values,acc_ar = get_max_pred_values(test_x,test_y)
            auc = get_auc_val(max_pred_values,acc_ar)[0]
    
            print("auc = " + str(auc))
        except:
            pass

#********************************************************************  
            
        
        
#****************************************
#**** data preparation and training for advanced confidence prediciton method
#****************************************
def get_max_pred_values_advanced(data):
    data_x,data_label,data_lmd,single_out,data_para = prep_data_advanced(data,0.1)
    res = model.predict(x=[data_x,data_label,data_lmd])
    confidences = res[0][:,NUM_CLASSES]
    predictions = res[0][:,0:NUM_CLASSES]
    pred_classes = np.argmax(predictions,axis=1)
    pred_classes = np.reshape(pred_classes,(pred_classes.shape[0],1))
    true_classes = np.argmax(data_label,axis=1)
    #acc_ar if entries are 0, accurate prediction
    acc_ar = pred_classes - true_classes
    acc_ar = (acc_ar == 0 )
    acc_ar=np.reshape(acc_ar,(acc_ar.shape[0],))
    return confidences,acc_ar

def prep_data_advanced(data,lmd):
    data_y = np.array([np.reshape(item[0],(NUM_CLASSES,)) for item in data])
    data_x = np.array([item[1] for item in data])
    data_label = np.reshape(data_y,(data_y.shape[0],NUM_CLASSES,1))
    data_lmd = np.full((data_x.shape[0],1,1),lmd)
    data_para = np.full((data_x.shape[0],3),0.1)
    conf_ar = np.zeros((data_label.shape[0],1,1))
    single_out = np.append(arr = data_label,values = conf_ar,axis=1)
    single_out = np.reshape(single_out,(single_out.shape[0],8))
    return data_x,data_label,data_lmd,single_out,data_para

def train_advanced():
    beta = 0.5
    lmd=0.2
    train,vali,test = get_train_test([0.6,0.8],extracted)
    train_x,train_label,train_lmd,train_single_out,train_para = prep_data_advanced(train,lmd)
    vali_x,vali_label,vali_lmd,vali_single_out,vali_para = prep_data_advanced(vali,lmd)
    for i in range(0,1000):
        res = model.fit(x=[train_x,train_label,train_lmd],y=[train_single_out,train_para],validation_data = [[vali_x,vali_label,vali_lmd],[vali_single_out,vali_para]],batch_size=64)
        val_acc = res.history['val_concatenate_1_accuracy_without_conf'][0]
        res = model.predict(x=[train_x,train_label,train_lmd])
        mean_res = np.mean(res[1],axis=0)
        L = mean_res[0]
        L_t = mean_res[1]
        L_c = mean_res[2]
    
        print(L_c)
        print(train_lmd[0][0][0])
        if(L_c>beta):
            lmd+=0.01
        else:
            lmd-=0.01
            
        if(val_acc > 0.4):
            K.set_value(model.optimizer.lr,0.0001)
            print(K.get_value(model.optimizer.lr))
        elif(val_acc > 0.5):
            K.set_value(model.optimizer.lr,0.000001)
            print(K.get_value(model.optimizer.lr))
    

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

#**********************************************
#*********Evaluate vanila confiednce prediction method
#**********************************************
import mini_xception as xception
importlib.reload(xception)    
model = xception.get_xception()

train_vanila()

max_pred_values,acc_ar = get_max_pred_values(test_x,test_y)
auc,points = get_auc_val(max_pred_values,acc_ar)
print("auc = " + str(auc))
plot_ROC(points)



#**********************************************
#*********Evaluate advanced confiednce prediction method
#**********************************************
import mini_xception as xception
importlib.reload(xception)    
model = xception.get_xception_conf()
train_advanced()


confidences,acc_ar = get_max_pred_values_advanced(test)
auc,points = get_auc_val(confidences,acc_ar)
print("auc = " + str(auc))
plot_ROC(points)



 
    
