# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:42:03 2019

@author: vasum
"""


#%matplotlib inline
from __future__ import print_function
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
import keras
from keras.models import load_model


df_final_test = pd.read_csv("Dataset/sensorlab_2019-03-01-02.06.17/accelerometer.csv")

# Same labels will be reused throughout the program
LABELS = ['Downstairs',
          'Jogging',
          'Sitting',
          'Standing',
          'Upstairs',
          'Walking']

df_final_test["activity"] = "Walking"
df_final_test["ActivityEncoded"] = 5


# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40
#--------------------------Reshape Data into Segments and Prepare for Keras--------------------------------------


pd.options.mode.chained_assignment = None  # default='warn'
df_final_test['x'] = df_final_test['x'] / df_final_test['x'].max()
df_final_test['y'] = df_final_test['y'] / df_final_test['y'].max()
df_final_test['z'] = df_final_test['z'] / df_final_test['z'].max()
# Round numbers
df_final_test = df_final_test.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})


def create_segments_and_labels(df_final_test, time_steps, step, label_name):

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df_final_test) - time_steps, step):
        xs = df_final_test['x'].values[i: i + time_steps]
        ys = df_final_test['y'].values[i: i + time_steps]
        zs = df_final_test['z'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df_final_test[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


x_test_final, y_test_final = create_segments_and_labels(df_final_test,
                                              TIME_PERIODS,
                                              10,
                                              'ActivityEncoded')

print('x_train shape: ', x_test_final.shape)
print(x_test_final.shape[0], 'training samples')
print('y_train shape: ', y_test_final.shape)


# Set input & output dimensions
num_time_periods, num_sensors = x_test_final.shape[1], x_test_final.shape[2]
num_classes = 6
#print(list(le.classes_))



input_shape = (num_time_periods*num_sensors)
x_test_final = x_test_final.reshape(x_test_final.shape[0], input_shape)
print('x_train shape:', x_test_final.shape)
print('input_shape:', input_shape)


x_test_final = x_test_final.astype('float32')
#y_test_final = y_test_final.astype('float32')

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

#model_m = load_model('ANN/Model/v3/best_model.0.88-03-0.74.h5')
model_m = load_model('best_model.0.77-0.82-12-0.71.h5')
y_pred_test_final = model_m.predict(x_test_final)
# Take the class with the highest probability from the test predictions
max_y_pred_test_final = np.argmax(y_pred_test_final, axis=1)

#Append dummy values for heatmap
max_y_pred_test_final = np.append(max_y_pred_test_final,[0])
y_test_final = np.append(y_test_final,[0])

max_y_pred_test_final = np.append(max_y_pred_test_final,[1])
y_test_final = np.append(y_test_final,[0])

max_y_pred_test_final = np.append(max_y_pred_test_final,[2])
y_test_final = np.append(y_test_final,[0])

max_y_pred_test_final = np.append(max_y_pred_test_final,[3])
y_test_final = np.append(y_test_final,[0])

max_y_pred_test_final = np.append(max_y_pred_test_final,[4])
y_test_final = np.append(y_test_final,[0])

max_y_pred_test_final = np.append(max_y_pred_test_final,[5])
y_test_final = np.append(y_test_final,[0])

# =============================================================================
# matrix = metrics.confusion_matrix(y_test_final, max_y_pred_test_final)
# print(matrix)
# =============================================================================
show_confusion_matrix(y_test_final, max_y_pred_test_final)

print(classification_report(y_test_final, max_y_pred_test_final))