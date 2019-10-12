# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:16:39 2019

@author: vasum
"""
#%matplotlib inline
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
import keras, pickle
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import os
from sklearn.model_selection import train_test_split

PICKLE_DIR = './pickle_wisdom/'

def read_data(file_path):

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
    df.dropna(axis=0, how='any', inplace=True)

    return df

def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan


def create_segments_and_labels(df, time_steps, step, label_name):

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def window_testing(TIME_PERIODS):
    
    print(f'For window size {TIME_PERIODS}')
    
    
    STEP_DISTANCE = TIME_PERIODS/2

    STEP_DISTANCE = int(STEP_DISTANCE)
    
    x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)
#    print('x_train shape: ', x_train.shape)
#    print(x_train.shape[0], 'training samples')
#    print('y_train shape: ', y_train.shape)
    
    
    # Set input & output dimensions
    num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
    num_classes = le.classes_.size
#    print(list(le.classes_))
    
    
    
    input_shape = (num_time_periods*num_sensors)
    x_train = x_train.reshape(x_train.shape[0], input_shape)
#    print('x_train shape:', x_train.shape)
#    print('input_shape:', input_shape)
    
    
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    
    
    y_train_hot = np_utils.to_categorical(y_train, num_classes)
#    print('New y_train shape: ', y_train_hot.shape)
    
    
    
    #---------------------------------------------------------------------------------------------------
    
    x_test, y_test = create_segments_and_labels(df_test,
                                                  TIME_PERIODS,
                                                  STEP_DISTANCE,
                                                  LABEL)
    
#    print('x_train shape: ', x_test.shape)
#    print(x_test.shape[0], 'training samples')
#    print('y_train shape: ', y_test.shape)
    
    
    # Set input & output dimensions
    num_time_periods, num_sensors = x_test.shape[1], x_test.shape[2]
    num_classes = le.classes_.size
#    print(list(le.classes_))
    
    
    
    input_shape = (num_time_periods*num_sensors)
    x_test = x_test.reshape(x_test.shape[0], input_shape)
#    print('x_train shape:', x_test.shape)
#    print('input_shape:', input_shape)
    
    
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    
    
    y_test_hot = np_utils.to_categorical(y_test, num_classes)
#    print('New y_train shape: ', y_test_hot.shape)
    
    #---------------------------------------------------------------------------------------------------
    
    
    #-------------------------Create Deep Neural Network Model in Keras-------------------------------------
    model_m = Sequential()
    # Remark: since coreml cannot accept vector shapes of complex shape like
    # [80,3] this workaround is used in order to reshape the vector internally
    # prior feeding it into the network
    model_m.add(Reshape((TIME_PERIODS, 3), input_shape=(input_shape,)))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Flatten())
    model_m.add(Dense(num_classes, activation='softmax'))
    print(model_m.summary())
    
    
    #-----------------------------------Fit the DNN Model in Keras-------------------------------------------
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.{val_acc:.2f}-{acc:.2f}-{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='acc', patience=1)
    ]
    
    model_m.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
    
    # Hyper-parameters
    BATCH_SIZE = 400
    EPOCHS = 50
    
    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    history = model_m.fit(x_train,
                          y_train_hot,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          callbacks=callbacks_list,
                          validation_split=0.2,
                          verbose=1)
    
    
    
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()
    
    # Print confusion matrix for training data
    y_pred_train = model_m.predict(x_train)
    # Take the class with the highest probability from the train predictions
    max_y_pred_train = np.argmax(y_pred_train, axis=1)
    print(classification_report(y_train, max_y_pred_train))
    
    
    
    
    y_pred_test = model_m.predict(x_test)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    
    show_confusion_matrix(y_test, max_y_pred_test)
    
    print(classification_report(y_test, max_y_pred_test))
    
    dump_path = PICKLE_DIR + str(TIME_PERIODS) + "/"
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    
    print("Dumping Performance records ..")
    pickle.dump(y_train, open(dump_path + "/y_train.p", "wb"))
    pickle.dump(max_y_pred_train, open(dump_path + "/max_y_pred_train.p", "wb"))
    pickle.dump(y_test, open(dump_path + "/y_test.p", "wb"))
    pickle.dump(max_y_pred_test, open(dump_path + "/max_y_pred_test.p", "wb"))
    
    print("\n---------------------------------------------------------\n")
    

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')

filepath = r'D:\ME\ME_2nd_Year\ADM\project\har\Dataset\final\dataset'

labels=[ name for name in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, name)) ]

#df = pd.read_csv("D:\ME\ME_2nd_Year\ADM\project\har\Dataset\colab_dataset_mobile.csv")

df = read_data('D:\ME\ME_2nd_Year\ADM\project\har\Dataset\WISDM_ar_v1.1_raw.txt')

# Define column name of the label vector
LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df['activity'].values.ravel())

#------------------------Normalize features for training data set (values between 0 and 1)-----------------------


df_test = df[df['user-id'] > 18]
df_train = df[df['user-id'] <= 18]


#------------------------Normalize features for training data set (values between 0 and 1)-----------------------
# This must also be done to testing set later
# Surpress warning for next 3 operation
pd.options.mode.chained_assignment = None  # default='warn'
df_train['x-axis'] = df_train['x-axis'] / df_train['x-axis'].max()
df_train['y-axis'] = df_train['y-axis'] / df_train['y-axis'].max()
df_train['z-axis'] = df_train['z-axis'] / df_train['z-axis'].max()
# Round numbers
df_train = df_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})



pd.options.mode.chained_assignment = None  # default='warn'
df_test['x-axis'] = df_test['x-axis'] / df_test['x-axis'].max()
df_test['y-axis'] = df_test['y-axis'] / df_test['y-axis'].max()
df_test['z-axis'] = df_test['z-axis'] / df_test['z-axis'].max()
# Round numbers
df_test = df_test.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})


windows =   [50, 80, 120, 160, 200]

for win in windows:
    window_testing(win)