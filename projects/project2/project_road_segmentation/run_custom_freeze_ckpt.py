import os, argparse

import tensorflow as tf
from tensorflow.core.framework import graph_pb2 as gpb
from google.protobuf import text_format as pbtf

STEP = 1177
input_graph_path = f"train_output/graph.pbtxt"
input_checkpoint_path = f"train_output/model.ckpt-{STEP}"
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
output_graph_path = f"frozen_model-{STEP}.pb"

def freeze_graph(model_dir, output_node_names):
    """
    model_dir: the root folder containing the checkpoint state file
    output_node_names: a string, containing all the output node's names, comma separated
    """
    # We retrieve our checkpoint fullpath
    print(model_dir)
    print(os.path.isfile(model_dir))
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    #absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = output_graph_path

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    with tf.Session(graph=tf.Graph()) as sess:
        with open('train_output/graph.pbtxt', 'r') as fp:
            graph_str = fp.read()

        graph_def = gpb.GraphDef()
        pbtf.Parse(graph_str, graph_def)

        tf.import_graph_def(graph_def)

        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        saver.restore(sess, input_checkpoint)

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), output_node_names.split(",")) 

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    freeze_graph("train_output", OUTPUT_TENSOR_NAME)
