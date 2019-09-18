#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:07:51 2017
This script is written to evaluate a pretrained model saved as  model.h5 using 'testData.npy' 
and 'groundTruth.npy'. This script reports the error as the cross entropy loss in percentage
and also generated a png file for the confusion matrix. 
@author:Muhammad Shahnawaz
"""
# importing libraries and dependecies 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

import tensorflow as tf
    

def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)


def segment_signal(df,N_TIME_STEPS,step):
        
    segments = []
    labels = []
    for i in range(0, len(df) - N_TIME_STEPS, step):
        ax = df["la_x"].values[i:i+N_TIME_STEPS]
        ay = df["la_y"].values[i:i+N_TIME_STEPS]
        az = df["la_z"].values[i:i+N_TIME_STEPS]
        fox = df["gx"].values[i:i+N_TIME_STEPS]
        foy = df["gy"].values[i:i+N_TIME_STEPS]
        foz = df["gz"].values[i:i+N_TIME_STEPS]
        label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
        segments.append([ax,ay,az,fox,foy,foz])
        labels.append(label)
    return segments, labels
  
  


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )

    input_name = graph.get_operations()[0].name+':0'
    output_name = graph.get_operations()[-1].name+':0'

    return graph, input_name, output_name


def predict(model_path, input_data):
    tf_model,tf_input,tf_output = load_graph(GRAPH_PB_PATH)
    
    # Create tensors for model input and output
    x = tf_model.get_tensor_by_name(tf_input)
    y = tf_model.get_tensor_by_name(tf_output) 

    # Number of model outputs      
    with tf.Session(graph=tf_model) as sess:
        y_out = sess.run(y, feed_dict={x: input_data})
        predictions = y_out

    return predictions

logs_dir = './logs'
#from comp_filter import Filter
#K.set_image_dim_ordering('th')
# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)

# matplotlib inline
plt.style.use('ggplot')
# defining function for loading the dataset

# The number of steps within one time segment
N_TIME_STEPS  = 200
# The steps to take from one segment to the next; if this value is equal to
# N_TIME_STEPS , then there is no overlap between the segments
step = 20

N_FEATURES = 6

plt.style.use('ggplot')
# =============================================================================
# # Reading the data
# df_test_1 = pd.read_csv("Dataset/Fused/v6/Fast/fused_2019_03_22_15_57_54.csv")
# df_test_1_1 = pd.read_csv("Dataset/Fused/v6/Fast/fused_2019_03_22_15_59_35.csv")
# #df_test_1_2 = pd.read_csv("Dataset/Fused/v3/fast/fused_2019_03_18_20_04_36.csv")
# df_test_1 = df_test_1.append(df_test_1_1)
# #df_test_1 = df_test_1.append(df_test_1_2)
# 
# df_test_2 = pd.read_csv("Dataset/Fused/v6/Normal/fused_2019_03_22_15_56_52.csv")
# df_test_2_1 = pd.read_csv("Dataset/Fused/v6/Normal/fused_2019_03_22_15_58_30.csv")
# #df_test_2_2 = pd.read_csv("Dataset/Fused/v3/normal/fused_2019_03_18_19_58_37.csv")
# df_test_2 = df_test_1.append(df_test_2_1)
# #df_test_2 = df_test_1.append(df_test_2_2)
# 
# df_test_3 = pd.read_csv("Dataset/Fused/v6/Slow/fused_2019_03_22_15_54_45.csv")
# #df_test_3_1 = pd.read_csv("Dataset/Fused/v3/slow/fused_2019_03_18_20_00_00.csv")
# #df_test_3_2 = pd.read_csv("Dataset/Fused/v3/slow/fused_2019_03_18_20_01_38.csv")
# #df_test_3 = df_test_1.append(df_test_3_1)
# #df_test_3 = df_test_1.append(df_test_3_2)
# 
# #df_4= pd.read_csv("Dataset/Fused/v1/slow/fused_2019_03_15_17_25_45.csv")
# #df_3= pd.read_csv("Dataset/walk_fast.csv")
# df_test_1["activity"] = "walk_fast"
# df_test_2["activity"] = "walk_normal"
# df_test_3["activity"] = "walk_slow"
# #df_1["ActivityEncoded"] = 1
# 
# 
# df_final_test = df_test_1
# #dataset = dataset.append(df_2)
# df_final_test = df_final_test.append(df_test_2)
# df_final_test = df_final_test.append(df_test_3)
# 
# # Define column name of the label vector
# LABEL = 'ActivityEncoded'
# # Transform the labels from String to Integer via LabelEncoder
# le = preprocessing.LabelEncoder()
# # Add a new column to the existing DataFrame with the encoded values
# df_final_test[LABEL] = le.fit_transform(df_final_test['activity'].values.ravel())
# 
# df_final_test.to_csv("colab_test.csv", index=False)
# =============================================================================
#df_2["activity"] = "walk_fast"
#df_3["activity"] = "walk_slow"
#df_4["activity"] = "walk_slow"

#dataset = df_final_test
#dataset = dataset.append(df_2)
#dataset = dataset.append(df_3)
#dataset = dataset.append(df_4)


#dataset.sort_values(by=['time'], inplace=True)
#
#dataset.dropna(axis=0, how='any', inplace= True)

# Same labels will be reused throughout the program
LABELS = ['walk_slow',
          'walk_fast',
          'walk_normal']


#X = pd.DataFrame(df_final_test[["ax","ay","az","fox","foy","foz"]])

# Round numbers

# Segmenting the test set
# Segmenting the signal in overlapping windows of 90 samples with 50% overlap
# =============================================================================
# 
# testset= pd.read_csv("Dataset/Test/Fast_Towards_Down/fused_2019_03_23_02_08_54.csv")  
# testset["activity"] = "walk_fast"
# testset_1= pd.read_csv("Dataset/Test/Normal_Towards_Down/fused_2019_03_23_02_07_45.csv")  
# testset_1["activity"] = "walk_normal"
# testset_2= pd.read_csv("Dataset/Test/Slow_Towards_Down/fused_2019_03_23_02_10_00.csv")  
# testset_2["activity"] = "walk_slow"
# 
# 
# testset = testset.append(testset_1)
# testset = testset.append(testset_2)
# 
# 
# X = pd.DataFrame(testset[["ax","ay","az","fox","foy","foz","la_x","la_y","la_z"]])
# X = feature_normalize(X)
# 
# testset[["ax","ay","az","fox","foy","foz","la_x","la_y","la_z"]] =  X
# testset = testset.round({"ax":4,"ay":4,"az":4,"fox":4,"foy":4,"foz":4,"la_x":4,"la_y":4,"la_z":4})
# 
# testset.to_csv("colab_v2_test.csv", index=False)
# =============================================================================

testset = pd.read_csv("colab_v2_test.csv")

test_segments, groundTruth = segment_signal(testset,N_TIME_STEPS ,step)

# Defining the class labels
#groundTruth[0] = "Standing"
#groundTruth[1] = "Upstairs"

#tf_model,tf_input, tf_output = load_graph(GRAPH_PB_PATH)

reshaped_test_segments = np.asarray(test_segments, dtype= np.float32).reshape(-1, N_TIME_STEPS , N_FEATURES)
groundTruth = np.asarray(pd.get_dummies(groundTruth), dtype = np.float32)

GRAPH_PB_PATH = './checkpoint/v5/hyperparam/v2.0/frozen_v2.0_har.pb'
#GRAPH_PB_PATH = './checkpoint/v4/la_fused/frozen_v3_har.pb'
   

y_pred = predict(GRAPH_PB_PATH, reshaped_test_segments)


# =============================================================================
# from keras.models import load_model
# model = load_model('./checkpoint/v4/keras_model_5.1.h5')
# y_pred = model.predict(reshaped_segments,verbose=2)
# =============================================================================

# Getting the class predicted and class in ground truth for creation of confusion matrix

LABELS = ['Walking Fast','Walking Normal' ,'Walking Slow']

max_test = np.argmax(groundTruth, axis=1)
max_predictions = np.argmax(y_pred, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)


plt.figure(figsize=(5, 3))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();


print(classification_report(max_test, max_predictions))





