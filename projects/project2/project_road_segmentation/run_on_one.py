
from PIL import Image

INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
INPUT_SIZE = 513
FROZEN_GRAPH_NAME = 'frozen_inference_graph'

def run(image):
    """ Runs inference on a single image """
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    with tf.Session(graph=load_graph()) as sess:
        batch_seg_map = sess.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

def load_graph(graphfile):
    """ Loads frozen graph file and returns a tf.Graph object """
    graph = tf.Graph()

    with open(graphfile) as fp:        
        graph_def = tf.GraphDef.FromString(fp.read())
        if graph_def is None:
            raise RuntimeError('Cannot load inference graph.')

    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    
    return graph

if __name__=="__main__":
    im = Image.open('datasets/roadseg/test_set_images/test_all/test_33.png')
    run(im)
