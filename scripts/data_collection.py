# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:40:17 2019

@author: vasum
"""

from read_wisdm_dataset import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os, pickle
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report


def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

def data_pre_processing(filepath, activity):
    filepath= filepath + "/" +  activity + '/**/*.csv'
    files =  glob(filepath)
    df = [pd.DataFrame(d[100:-100]) for d in [pd.read_csv(filename,sep=","  ) for filename in files]]
    start = 0
    for d in df:
        end = start + len(d)
        d['ind'] = np.arange(start,end)
        start=end
    df = pd.concat(df, axis=0)
    df["activity"] = activity
    return df

def read_dataset(filepath):
    parent = os.path.abspath(os.path.join(filepath, os.pardir))
    
    labels=[ name for name in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, name)) ]
    df = [data_pre_processing(filepath, label) for label in labels]
    df = pd.concat(df)
    
    df = df.drop(columns=['fox', 'foy', 'foz'])
    
    df.to_csv(parent + "/colab_dataset.csv", index=False)
    df= pd.read_csv(parent +"/colab_dataset.csv")
    
    df.dropna(axis=0, how='any', inplace= True)
    
    # Define column name of the label vector
    LABEL = 'ActivityEncoded'
    le = preprocessing.LabelEncoder()
    df[LABEL] = le.fit_transform(df['activity'].values.ravel())
    
    X = pd.DataFrame(df[["ax","ay","az","la_x","la_y","la_z"]])
    X = feature_normalize(X)
    
    df[["ax","ay","az","la_x","la_y","la_z"]] =  X
    df = df.round({"ax":4,"ay":4,"az":4,"la_x":4,"la_y":4,"la_z":4})
    return df

#-----------------------------------------Evaluation-------------------------------
def printCM(predictions ,y_test, LABELS):    
    max_test = np.argmax(y_test, axis=1)
    max_predictions = np.argmax(predictions, axis=1)
    confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show();
    print(classification_report(max_test, max_predictions))
    return max_test,max_predictions
    
    
def get_tuning_result(PICKLE_DIR):
    pickles = [ name for name in os.listdir(PICKLE_DIR) if os.path.isdir(os.path.join(PICKLE_DIR, name)) ]
    df = {}
    for f in pickles:
        filepath=PICKLE_DIR + f + '/performance_records.p'
        dump = pickle.load(open(filepath, 'rb'))
        params = [key for index, key in enumerate(dump)]
        report = [{'predictions' : val['predictions'], 'true_labels' : val['true_labels']}  for key, val in [(key, dump[key]) for key in params]]
        
        temp = [pd.DataFrame({key : val['score']}) for key, val in [(key, dump[key]) for key in params]]
        temp = pd.concat(temp, axis = 1)
        
        df[f] = {'params' : params, 'scores':report, 'acc' : temp}
    return df

def print_scores(report , key, index, LABELS):    
    print(f"\n------------- Report for Key : {key} -----------------")
    if index != None:
        item = report[key]['scores'][index]
        printCM(item['predictions'],item['true_labels'],LABELS)
    else:
        report = report[key]['scores']
        [printCM(item['predictions'],item['true_labels'],LABELS) for item in report]
        
def get_tuning_params(PICKLE_DIR, window):
    filepath=PICKLE_DIR + window + '/performance_records.p'
    dump = pickle.load(open(filepath, 'rb'))
    acc = 0
    for index, key in enumerate(dump):
        if(acc < dump[key]['score']['acc']):
            acc = dump[key]['score']['acc']
            param = key
        
#    dump[key1]['score']['acc']
#    params = [key for index, key in enumerate(dump)]
    return param

def get_wisdm_labels():
    return ['Downstairs','Jogging','Sitting','Standing','Upstairs','Walking']

def get_mobile_labels():
    filepath = 'D:/Bits/Sem 3/ADM/Project/python/har/Dataset/final/dataset'
    return [ name for name in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, name)) ]

def get_window_list():
    return [50, 80, 120, 160, 200]
#window_list  = [50, 80, 120, 160, 200]
#
#WISDM_PICKLE_DIR='./pickle/wisdm/'
#WISDM_LABELS = get_wisdm_labels()
#
#MOBILE_LABELS = get_mobile_labels()
#MOBILE_PICKLE_DIR='./pickle/mobile/'
#
#report = get_tuning_result(WISDM_PICKLE_DIR) 
#[print_scores(report, data, None,WISDM_LABELS) for data in report]