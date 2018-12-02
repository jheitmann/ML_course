
from tensorflow.python.tools import freeze_graph

checkpoint_state_name = "checkpoint_state"

STEP = 1166
input_graph_path = f"datasets/graph.pbtxt"
input_saver_def_path = ""
input_binary = False
input_checkpoint_path = f"datasets/model.ckpt-{STEP}"

INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
INPUT_SIZE = 513
FROZEN_GRAPH_NAME = 'frozen_inference_graph'

output_node_names = OUTPUT_TENSOR_NAME
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = f"frozen_model-{STEP}.pb"
clear_devices = False

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, input_checkpoint_path,
                          output_node_names, restore_op_name,
                          filename_tensor_name, output_graph_path, 
                          clear_devices)
