# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:23:46 2019

@author: vasum
"""
# importing libraries and dependecies 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


import tensorflow as tf
import pickle
import os
from glob import glob


logs_dir = './logs'
#from comp_filter import Filter
#K.set_image_dim_ordering('th')
# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)

# matplotlib inline
plt.style.use('ggplot')
# defining function for loading the dataset
#
# The number of steps within one time segment
N_TIME_STEPS  = 200
# The steps to take from one segment to the next; if this value is equal to
# N_TIME_STEPS , then there is no overlap between the segments
step = 20
#
N_FEATURES = 6
N_CLASSES = 3
#---------------------------------Dataset and Preprocessing---------------------------------------

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
    
def plot_activity(activity,df):
    data = df[df['activity'] == activity][['la_x', 'la_y', 'la_z']][:200] #[:13530]
#    data = df[['la_x', 'la_x', 'la_x']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12), title=activity)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(0.5, 1))
        
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
#        segments.append([ax,ay,az])
        segments.append([ax,ay,az,fox,foy,foz])
        labels.append(label)
    return segments, labels

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    
    
    
#dataset.to_csv("./Dataset/Fused/v2/colab_dataset.csv", index =False)
#dataset= pd.read_csv("Report/classify/Fast_Towards_Up/fused_2019_04_21_01_55_36.csv")
dataset = dataset.append(pd.read_csv("colab_v2.1.csv"))

#dataset2= pd.read_csv("colab_v3.csv")
plot_activity("walk_fast",dataset)
dataset['activity'].value_counts().plot(kind='bar', title='Training by Activity Type');
# =============================================================================
# fast_files = glob('Dataset/Left/Fast/**/*.csv')
# fast = [pd.DataFrame(df[100:-100]) for df in [pd.read_csv(filename,sep=","  ) for filename in fast_files]]
# fast = pd.concat(fast, axis=0)
# fast["activity"] = "walk_fast"
# 
# normal_files = glob('Dataset/Left/Normal/**/*.csv')
# normal = [pd.DataFrame(df[150:-100]) for df in [pd.read_csv(filename,sep=","  ) for filename in normal_files]]
# normal = pd.concat(normal, axis=0)
# #normal = normal.sample(n = 13530, replace = False)
# normal["activity"] = "walk_normal"
# normal.dropna(axis=0, how='any', inplace= True)
# 
# slow_files = glob('Dataset/Left/Slow/**/*.csv')
# slow = [pd.DataFrame(df[200:-100]) for df in [pd.read_csv(filename,sep=","  ) for filename in slow_files]]
# slow = pd.concat(slow, axis=0)
# #slow = slow.sample(n = 13530, replace = False) 
# #slow.sort_values(by=['time'], inplace=True)
# slow["activity"] = "walk_slow"
# slow.dropna(axis=0, how='any', inplace= True)
# 
# #fast_files = glob('Dataset/Right/Fast/**/*.csv')
# #fast = [pd.DataFrame(df)[100:-100] for df in [pd.read_csv(filename,sep=","  ) for filename in fast_files]]
# #fast = pd.concat(fast, axis=0)
# #fast["activity"] = "walk_fast"
# #
# #normal_files = glob('Dataset/Right/Normal/**/*.csv')
# #normal = [pd.DataFrame(df)[100:-100] for df in [pd.read_csv(filename,sep=","  ) for filename in normal_files]]
# #normal = pd.concat(normal, axis=0)
# #normal = normal.sample(n = 13530, replace = False) 
# #normal.sort_values(by=['time'], inplace=True)
# #normal["activity"] = "walk_normal"
# #
# #slow_files = glob('Dataset/Right/Slow/**/*.csv')
# #slow = [pd.DataFrame(df)[100:-100] for df in [pd.read_csv(filename,sep=","  ) for filename in slow_files]]
# #slow = pd.concat(slow, axis=0)
# #slow = slow.sample(n = 13530, replace = False) 
# #slow.sort_values(by=['time'], inplace=True)
# #slow["activity"] = "walk_slow"
# 
# 
# dataset = fast
# dataset = dataset.append(normal)
# dataset = dataset.append(slow)
# 
# dataset.sort_values(by=['time'], inplace=True)
# dataset.dropna(axis=0, how='any', inplace= True)
# 
# # Define column name of the label vector
# LABEL = 'ActivityEncoded'
# # Transform the labels from String to Integer via LabelEncoder
# le = preprocessing.LabelEncoder()
# # Add a new column to the existing DataFrame with the encoded values
# dataset[LABEL] = le.fit_transform(dataset['activity'].values.ravel())
# #
# 
# X = pd.DataFrame(dataset[["ax","ay","az","fox","foy","foz","la_x","la_y","la_z"]])
# X = feature_normalize(X)
# 
# dataset[["ax","ay","az","fox","foy","foz","la_x","la_y","la_z"]] =  X
# dataset = dataset.round({"ax":4,"ay":4,"az":4,"fox":4,"foy":4,"foz":4,"la_x":4,"la_y":4,"la_z":4})
# =============================================================================

#dataset.to_csv("colab_v2.1.csv", index = False)
#
#

# Segmenting the signal in overlapping windows of 90 samples with 50% overlap
segments, labels = segment_signal(dataset,N_TIME_STEPS ,step)

np.array(segments).shape


reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS , N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=0)


# Segmenting the test set
# Segmenting the signal in overlapping windows of 90 samples with 50% overlap

testset= pd.read_csv("colab_v2_test.csv")  
# =============================================================================
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

test_segments, test_labels = segment_signal(testset,N_TIME_STEPS ,step)

np.array(segments).shape
#
#test_labels[0] = 0
#test_labels[2] = 2
test_reshaped_segments = np.asarray(test_segments, dtype= np.float32).reshape(-1, N_TIME_STEPS , N_FEATURES)
test_labels = np.asarray(pd.get_dummies(test_labels), dtype = np.float32)
#-----------------------------Creating The Model-------------------------------------------



# =============================================================================
# from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape,LSTM
# import keras 
# from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier 
# from keras.models import Sequential
# trainSplitRatio = 0.8
# trainSplit = np.random.rand(len(segments)) < trainSplitRatio
# 
# def cnnModel():
#     
#     model = Sequential()
#     model.add(Dense(N_HIDDEN_UNITS, activation='relu',input_shape=( N_TIME_STEPS, N_FEATURES)))
#     model.add(Dense(N_HIDDEN_UNITS, activation='relu'))
# #    model.add(Dropout(dropOutRatio))  
#     model.add(LSTM(N_HIDDEN_UNITS, return_sequences=True)) 
#     model.add(LSTM(N_HIDDEN_UNITS))
#     model.add(Dense(N_CLASSES, activation='softmax'))
#             
#     adam = optimizers.Adam(lr = 0.001, decay=1e-6)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#     return model
# 
# 
# model = cnnModel()
# 
# model.summary()
# 
# 
# #-----------------------------------Fit the DNN Model in Keras-------------------------------------------
# callbacks_list = [
#     keras.callbacks.ModelCheckpoint(
#         filepath='best_model.{val_acc:.2f}-{acc:.2f}-{epoch:02d}-{val_loss:.2f}.h5',
#         monitor='val_loss', save_best_only=True),
#     keras.callbacks.EarlyStopping(monitor='acc', patience=2)
# ]
# #metrics=[tf.keras.metrics.categorical_accuracy]
# 
# 
# history = model.fit(X_test,y_test, validation_split=1-trainSplitRatio,epochs=N_EPOCHS,
#                      batch_size=BATCH_SIZE,verbose=1)
# model.save('keras_model_5.1.h5')
# =============================================================================
 
# Our model contains 2 fully-connected and 2 LSTM layers (stacked on each other) with 64 units each:
def model(inputs, batch_size, N_HIDDEN_UNITS):
    weights = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    
    bias = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    with tf.name_scope('inputs'):
        X = tf.transpose(inputs, [1, 0, 2])
        X = tf.reshape(X, [-1, N_FEATURES])
    
    with tf.name_scope('hidden'):
        hidden = tf.nn.relu(tf.matmul(X, weights['hidden']) + bias['hidden'])  
        hidden = tf.split(hidden, N_TIME_STEPS, 0)
    tf.summary.histogram("hidden", hidden)  


    # Stack 2 LSTM layers
    lstm = [tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for num in range(2)]
    lstm = tf.nn.rnn_cell.MultiRNNCell(lstm)
    
    ### Run the data through the RNN layers
    with tf.name_scope("RNN_forward"):
    # Run each sequence step through the RNN with tf.nn.dynamic_rnn 
        outputs, _ = tf.nn.static_rnn(lstm, hidden, dtype=tf.float32)          

    # Get output for the last time step
    final_layer = outputs[-1]
    tf.summary.histogram("final_layer", final_layer)  
#    tf.summary.histogram('final_layer', final_layer)

    return tf.matmul(final_layer, weights['output']) + bias['output']



def run_model(N_TIME_STEPS,BATCH_SIZE, LEARNING_RATE,N_EPOCHS, N_HIDDEN_UNITS,L2_LOSS, log_type,train_writer,
              test_writer,validation_writer,early_stopping_step):
    #placeholders for our model:

    best_acc = 0
    stopping_step =0;
    tf.reset_default_graph()
    
    X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
    Y = tf.placeholder(tf.float32, [None, N_CLASSES])
    
    # Note that we named the input tensor, that will be useful when using the model from Android. Creating the model:
    pred_Y = model(X,BATCH_SIZE, N_HIDDEN_UNITS)
    tf.summary.histogram("matmul", pred_Y) 
    
    pred_softmax = tf.nn.softmax(pred_Y, name="y_")
    tf.summary.histogram("pred_softmax", pred_softmax)  
    
    
#    # Again, we must properly name the tensor from which we will obtain predictions. We will use L2 regularization and that must be noted in our loss op:
#    L2_LOSS = 0.0015
    
    l2 = L2_LOSS * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_Y, labels = Y)) + l2
    tf.summary.scalar('loss', loss)
    
    
    #optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    with tf.name_scope('train'):
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    
    
    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    saver = tf.train.Saver()
    
    history = dict(train_loss=[], 
                         train_acc=[], 
                         test_loss=[], 
                         test_acc=[])
    
    sess=tf.InteractiveSession()
    merged = tf.summary.merge_all()
    
    
    sess.run(tf.global_variables_initializer())
    
    
    train_count = len(X_train)
    
    for i in range(1, N_EPOCHS + 1):
        for start, end in zip(range(0, train_count, BATCH_SIZE),
                              range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
            sess.run([optimizer], feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})
    
        _, acc_train, loss_train,summary = sess.run([pred_softmax, accuracy, loss,merged], feed_dict={
                                                X: X_train, Y: y_train})
            
        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)
        
#        train_writer.add_summary(summary, i)
        if (acc_train > best_acc):
            stopping_step = 0
            best_acc = acc_train
            print("Epoch: {} acc:{}".format(i,acc_train))
        else:
            stopping_step += 1
            print("Epoch: {} acc:{}".format(i,acc_train))
        if stopping_step >= early_stopping_step:
            print("Early stopping is trigger at step: {} acc:{}".format(i,acc_train))
            break;
    
    
        if i != 1 and i % 10 != 0:
            continue
    
        _, acc_test, loss_test, summary = sess.run([pred_softmax, accuracy, loss,merged], feed_dict={
                                                X: X_test, Y: y_test})
#        test_writer.add_summary(summary, i)
    
        history['test_loss'].append(loss_test)
        history['test_acc'].append(acc_test)
        print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test} | train accuracy: {acc_train} loss :{loss_train}')
        
        
    print("Flushing writer data..")
    
    
    predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})
    
    print()
    print(f'final results: accuracy: {acc_final} loss: {loss_final}')
    
    
    #Validation with test data
    pred , acc_test, loss_test, summary = sess.run([pred_softmax, accuracy, loss,merged], feed_dict={X: test_reshaped_segments, Y: test_labels})
    printCM(pred, test_labels)
#    validation_writer.add_summary(summary, i)
    
    
#    train_writer.flush()
#    test_writer.flush()
#    validation_writer.flush()
#    
#    train_writer.close()
#    test_writer.close()
#    validation_writer.close()
    
    return history, predictions, sess, acc_final, loss_final, saver

#tf.InteractiveSession.close(sess)
#performance_records[(epochs, hidden_units)] = { 'acc_final' : acc_final, 'loss_final' : loss_final}

#-----------------------------------------Evaluation-------------------------------
def printCM(predictions ,y_test):

  LABELS = ['Walking Fast', 'Walking Normal', 'Walking Slow']

  max_test = np.argmax(y_test, axis=1)
  max_predictions = np.argmax(predictions, axis=1)
  confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

  plt.figure(figsize=(5, 3))
  sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
  plt.title("Confusion matrix")
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show();

  print(classification_report(max_test, max_predictions))

#0.00031779567496887653 78 40 65
    
L2_LOSS = 0.0015

N_HIDDEN_UNITS = 64

N_EPOCHS = 50
# should ideally be  above 1000 say 1024
BATCH_SIZE = 326

#  define optimizer and accuracy ops:
LEARNING_RATE = 0.001527643691277451
#
#
#N_HIDDEN_UNITS = 64
#
#N_EPOCHS = 50
## should ideally be  above 1000 say 1024
#BATCH_SIZE = 1024
#
##  define optimizer and accuracy ops:
#LEARNING_RATE = 0.000389


# =============================================================================
# N_HIDDEN_UNITS = 64
#  
# N_EPOCHS = 50
#  # should ideally be  above 1000 say 1024
# BATCH_SIZE = 100
#  
# #  define optimizer and accuracy ops:
# LEARNING_RATE = 0.0025
# =============================================================================
# Path to be created
trainPath = "./logs/" + str("test") + "/train/" 
testPath = "./logs/" +str("test") + "/test/" 
validationPath = "./logs/" + str("test") + "/validation/"
if not os.path.exists(trainPath):
    os.makedirs(trainPath)
if not os.path.exists(testPath):
    os.makedirs(testPath)
if not os.path.exists(testPath):
    os.makedirs(testPath)


train_writer = tf.summary.FileWriter(trainPath)
test_writer = tf.summary.FileWriter(testPath)
validation_writer = tf.summary.FileWriter(validationPath)

history, predictions, sess, acc_final, loss_final, saver = run_model(N_TIME_STEPS,BATCH_SIZE, LEARNING_RATE,N_EPOCHS, 
                                                                     N_HIDDEN_UNITS,L2_LOSS, "epochs_hu",train_writer,
                                                                     test_writer,validation_writer,3)

#p = sess.run(test_reshaped_segments)
tf.InteractiveSession.close(sess)
#history, predictions, sess, acc_final, loss_final, saver = run_model(N_TIME_STEPS,70, 0.00337,72, 42, 
#                                                                     "final_test",None,None)
#tf.InteractiveSession.close(sess)
#--------------------------------- Parameter tuning -----------------------------------
# =============================================================================
# performance_records = {}
# def generate_random_hyperparams(lr_min, lr_max, kp_min, kp_max, ep_min, ep_max):
#     '''generate random learning rate and keep probability'''
#     # random search through log space for learning rate
#     random_learning_rate = 10**np.random.uniform(lr_min, lr_max)
#     random_keep_prob = np.random.uniform(kp_min, kp_max)
#     epochs = np.random.uniform(ep_min, ep_max)
#     return random_learning_rate, int(random_keep_prob), int(epochs)
# 
# for i in range(1): # random search hyper-parameter space 20 times
#     print("==============================================================================================")
#     random_learning_rate, random_batch_size, epochs = generate_random_hyperparams(-5, -1, 50, 250, 30, 80)  
#     print(random_learning_rate, random_batch_size, epochs)    
#     history, predictions, sess, acc_final, loss_final, saver = run_model(N_TIME_STEPS,random_batch_size, 
#                                                                          round(random_learning_rate,6),100, 64)
#     tf.InteractiveSession.close(sess)
#     performance_records[(random_learning_rate, random_batch_size)] = { 'acc_final' : acc_final, 'loss_final' : loss_final}
# =============================================================================



# =============================================================================
# N_EPOCHS = 20
# history, predictions, sess, acc_final, loss_final, saver = run_model(N_TIME_STEPS,BATCH_SIZE, LEARNING_RATE, N_EPOCHS)
# tf.InteractiveSession.close(sess)
# =============================================================================

#-----------------------------------------Evaluation-------------------------------
plt.figure(figsize=(12, 8))

plt.plot(np.array(history['train_loss']), "r--", label="Train loss")
plt.plot(np.array(history['train_acc']), "g--", label="Train accuracy")

plt.plot(np.array(history['test_loss']), "r-", label="Test loss")
plt.plot(np.array(history['test_acc']), "g-", label="Test accuracy")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training Epoch')
plt.ylim(0)

plt.show()


LABELS = ['Walking Fast', 'Walking Normal', 'Walking Slow']

max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

plt.figure(figsize=(5, 3))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();

print(classification_report(max_test, max_predictions))
#------------------------------------Exporting the model------------------------------------

# Storing model to disk
DUMP_DIR = './checkpoint/v5/hyperparam/v2.0/'

if not os.path.exists(DUMP_DIR):
    os.makedirs(DUMP_DIR)
pickle.dump(predictions, open(DUMP_DIR + "predictions.p", "wb"))
pickle.dump(history, open(DUMP_DIR + "history.p", "wb"))
tf.train.write_graph(sess.graph_def, '.', DUMP_DIR + 'har.pbtxt')  
saver.save(sess, save_path = DUMP_DIR + "har.ckpt")
sess.close()

from tensorflow.python.tools import freeze_graph

MODEL_NAME = 'har'

input_graph_path =  DUMP_DIR +  MODEL_NAME +'.pbtxt'
checkpoint_path =  DUMP_DIR + MODEL_NAME + '.ckpt'
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name =DUMP_DIR + 'frozen_v2.0_'+MODEL_NAME+'.pb'

freeze_graph.freeze_graph(input_graph_path, input_saver="",
                          input_binary=False, input_checkpoint=checkpoint_path, 
                          output_node_names="y_", restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0", 
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")
    

#--------------------------------Verify tensor names------------------------------------

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and return it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph('./checkpoint/v3/frozen_v3_'+MODEL_NAME+'.pb')

for op in graph.get_operations(): 
        if "y_" in op.name or "input" in op.name:
            print(op.name, " : " ,op.values())