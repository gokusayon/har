# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:07:23 2019

@author: vasum
"""
from keras import backend as K
import tensorflow as tf
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

from keras.models import load_model

file_path = "v5-1"
model = load_model(file_path + ".h5")
print(model.outputs)
# [<tf.Tensor 'dense_2/Softmax:0' shape=(?, 10) dtype=float32>]
print(model.inputs)

# serialize model to JSON
# =============================================================================
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# =============================================================================
# [<tf.Tensor 'conv2d_1_input:0' shape=(?, 28, 28, 1) dtype=float32>]

# Saving the Model as tf
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph
    
# =============================================================================
#         print("\nORIGINAL GRAPH DEF Ops ===========================================")
#         ops = session.graph.get_operations()
#         for op in ops:
#             if "Softmax" in op.name or "input" in op.name:
#                 print([op.name, op.values()])
#         # save original graphdef to text file
#         with open("estimator_graph.pbtxt", "w") as fp:
#             fp.write(str(session.graph_def))
# =============================================================================

# =============================================================================
#         print("\nFROZEN GRAPH DEF Nodes ===========================================")
#         for node in frozen_graph.node:
#             print(node.name)
#         # save frozen graph def to text file
#         with open("estimator_frozen_graph.pbtxt", "w") as fp:
#             fp.write(str(frozen_graph))
# =============================================================================

#        print(freeze_var_names)

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph,"model/", file_path+".pb", as_text=False)
 
# Converting a tf.keras model into tflite
converter = tf.lite.TFLiteConverter.from_keras_model_file(file_path + ".h5" )
tflite_model = converter.convert()
open(file_path + ".tflite", "wb").write(tflite_model)

#
#tf.saved_model.simple_save(K.get_session(),"model/",inputs={"conv2d_1_input_1": model.inputs}, outputs={"dense_3/Softmax": model.outputs})
#[out.op.name for out in model.outputs]
#[out.op.name for out in model.inputs]
converter = tf.contrib.lite.TocoConverter.from_frozen_graph(frozen_graph, ["conv2d_1_input"], ["dense_3/Softmax"])
tflite_model = converter.convert()
open("fina_graph.tflite", "wb").write(tflite_model)
