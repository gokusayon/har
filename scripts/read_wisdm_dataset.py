#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:38:39 2019

@author: vasum
"""

# importing libraries and dependecies 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#K.set_image_dim_ordering('th')

# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)

# matplotlib inline
plt.style.use('ggplot')

#---------------------------------Dataset and Preprocessing---------------------------------------
def read_wisdm_data(file_path):

    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    df['z-axis'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')
    # ... and then this column must be transformed to float explicitly
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    
    df.dropna(axis=0, how='any', inplace= True)
    
    #------------------------Normalize features for training data set (values between 0 and 1)-----------------------
    # This must also be done to testing set later
    pd.options.mode.chained_assignment = None  # default='warn'
    df['x-axis'] = df['x-axis'] / df['x-axis'].max()
    df['y-axis'] = df['y-axis'] / df['y-axis'].max()
    df['z-axis'] = df['z-axis'] / df['z-axis'].max()

    return df

def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan
 
def show_basic_dataframe_info(dataframe):

    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))


def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma
    
def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
    
def plot_activity(activity,data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (15, 10), sharex = True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
    
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
        x = data["x-axis"][i:i+window_size]
        y = data["y-axis"][i:i+window_size]
        z = data["z-axis"][i:i+window_size]
#        if(len(dataset['timestamp'][i:i+window_size] == window_size):
        segments = np.vstack([segments,np.dstack([x,y,z])])
        labels = np.append(labels,stats.mode(data["activity"][i:i+window_size])[0][0])
#        print(".", end="")
#        print(i,".", i+window_size)
        
    return segments, labels

#------------------------------reading the data------------------------------------------
    


#-----------------------------------Validating Results-----------------------------------------------------
def get_labels():
    LABELS = ['Downstairs','Jogging','Sitting','Standing','Upstairs','Walking']
    return LABELS

