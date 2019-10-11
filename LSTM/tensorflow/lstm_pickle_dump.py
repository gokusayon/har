# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:48:14 2019

@author: vasum
"""
import pickle,os
import pandas as pd
window_list  = [50, 80, 120, 160, 200]

filepath = "pickle"
dump = pickle.load(open('pickle\score.p', 'rb'))

df = [ pd.DataFrame({d : data['score']}) for d, data in enumerate(dump)]
df = pd.concat(df, axis=1)

pickle.dump(df, open('pickle\df.p','wb'))

#0 -- 5  batch_size 62, epochs 10, hidden_units 68, l2_loss 0.002477993485288549, learning_rate 0.0027111720819702986
# {'acc': 0.87262356, 'loss': 1.4284681}

#1 -- 1  batch_size 413, epochs 43, hidden_units 64, l2_loss 0.0021748704621434678, learning_rate 0.0025271497924369904
# {'acc': 0.8966565, 'loss': 1.4916358}
 
#2 -- 3  batch_size 400, epochs 30, hidden_units 78, l2_loss 0.0011750556181198664, learning_rate 0.0011750556181198664
# {'acc': 0.89041096, 'loss': 1.7156473}
 
#3 -- 4  batch_size 229, epochs 15, hidden_units 68, l2_loss 0.001667240515268304, learning_rate 0.002415977827566457
# {'acc': 0.847561, 'loss': 1.6858381}
 
#4 -- 4  batch_size 229, epochs 15, hidden_units 68, l2_loss 0.001667240515268304, learning_rate 0.002415977827566457
#  {'acc': 0.82575756, 'loss': 1.5569212}
 
