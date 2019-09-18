#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:07:51 2017
This script is written to evaluate a pretrained model saved as  model.h5 using 'testData.npy' 
and 'groundTruth.npy'. This script reports the error as the cross entropy loss in percentage
and also generated a png file for the confusion matrix. 
@author:Muhammad Shahnawaz
"""
# importing the dependencies
from keras.models import load_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report

# Reading the data
df_final_test = pd.read_csv("Dataset/sensorlab_2019-03-01-05.27.22/accelerometer.csv")

# Same labels will be reused throughout the program
LABELS = ['Downstairs',
          'Jogging',
          'Sitting',
          'Standing',
          'Upstairs',
          'Walking']

df_final_test["activity"] = "Walking"
df_final_test["ActivityEncoded"] = 5

df_final_test['x'] = df_final_test['x'] / df_final_test['x'].max()
df_final_test['y'] = df_final_test['y'] / df_final_test['y'].max()
df_final_test['z'] = df_final_test['z'] / df_final_test['z'].max()
# Round numbers
df_final_test = df_final_test.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})


def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)

def segment_signal(data,window_size = 90,step = 45):
    segments = np.empty((0,window_size,3))
    labels = np.empty((0))
    for i in range(0, len(data) - window_size, step):
#    for (start, end) in windows(data['timestamp'], window_size):
        x = data["x"][i:i+window_size]
        y = data["y"][i:i+window_size]
        z = data["z"][i:i+window_size]
#        if(len(dataset['timestamp'][i:i+window_size] == window_size):
        segments = np.vstack([segments,np.dstack([x,y,z])])
        labels = np.append(labels,stats.mode(data["activity"][i:i+window_size])[0][0])
        
    return segments, labels

test_x, groundTruth = segment_signal(df_final_test,
                                              90,
                                              45)

# Defining the class labels
groundTruth[5] = "Downstairs"
groundTruth[1] = "Jogging"
groundTruth[2] = "Sitting"
groundTruth[3] = "Standing"
groundTruth[4] = "Upstairs"

test_x = test_x.reshape(test_x.shape[0], 90, 3,1)
groundTruth = np.asarray(pd.get_dummies(groundTruth),dtype = np.int8)

# Loading the pretrained model
#model = load_model('CNN/Models/v1.1/best_model.0.76-0.86-05-0.68.h5')
model = load_model('best_model.0.35-0.80-03-2.04.h5')
#loading the testData and groundTruth data
'''
 Creating and plotting a confusion matrix
'''
# predicting the classes
predictions = model.predict(test_x,verbose=2)

# Evaluating the model
score = model.evaluate(test_x,groundTruth,verbose=2)
print('Baseline Error: %.2f%%' %(100-score[1]*100))

# Getting the class predicted and class in ground truth for creation of confusion matrix
predictedClass = np.zeros((predictions.shape[0]))
groundTruthClass = np.zeros((groundTruth.shape[0]))
for instance in range (groundTruth.shape[0]):
    predictedClass[instance] = np.argmax(predictions[instance,:])
    groundTruthClass[instance] = np.argmax(groundTruth[instance,:])
    

# ------------------------------Validating the Results-----------------------------------------
# plotting the confusion matrix
def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
show_confusion_matrix(groundTruthClass,predictedClass)
print(classification_report(groundTruthClass, predictedClass))




