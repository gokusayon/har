# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:40:17 2019

@author: vasum
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from glob import glob


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