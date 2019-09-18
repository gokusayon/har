# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:35:28 2019

@author: vasum
"""



#-------------------------------------Make Predictions--------------------------------------------------

import tensorflow as tf
from tensorflow.python.platform import gfile
# =============================================================================
# 
# graph_def = tf.GraphDef()
# # Parses a serialized binary message into the current message.
# graph_def.ParseFromString(open('model/tf_model_1.pb','rb').read())
#     
#     [n.name + '=>' +  n.op for n in graph_def.node ]
# 
# [n.name + '=>' +  n.op for n in graph_def.node ]
# [n.name + '=>' +  n.op for n in graph_def.node if n.op in ( 'Softmax','Mul')]
# tensor_names = [t.name for op in tf.get_default_graph().get_operations() for t in op.values()]
# 
# 
# def printTensors(pb_file):
# 
#     # read pb into graph_def
#     with tf.gfile.GFile(pb_file, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
# 
#     # import graph_def
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def)
# 
#     # print operations
#     for op in graph.get_operations():
#         print(op.name)
# 
# 
# printTensors("model/tf_model.pb")
# =============================================================================
tensor_names = []
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

graph = load_graph("model/v5-1/" +"v5-1.pb")

for op in graph.get_operations(): 
        if "reshape" in op.name or "Softmax" in op.name or "conv" in op.name:
            print(op.name, " : " ,op.values())


