
import sys
# All those path needed in pythonpath : research/, research/slim, research/deeplab/datasets
#sys.path.append("./models/research/")
#sys.path.append("./models/research/slim/")
sys.path.append("C:\\Users\\gabjan\\Documents\\models\\research\\deeplab\\datasets")
from deeplab.datasets import build_voc2012_data

VOC_ROOT = "datasets/voc2012"
PROJ_ROOT = "datasets/roadseg"
IMAGE_FOLDER_VOC = f"{VOC_ROOT}/JPEGImages/"
LIST_FOLDER_VOC = f"{VOC_ROOT}/ImageSets/Segmentation/"
SEMANTIC_SEG_FOLDER_VOC = f"{VOC_ROOT}/SegmentationClass/"
IMAGE_FOLDER_PROJ = f"{PROJ_ROOT}/images"
LIST_FOLDER_PROJ = f"{PROJ_ROOT}/groundtruth"

# Directory containing original images
IMAGE_FOLDER = IMAGE_FOLDER_VOC
# Directory containing classification masks
LIST_FOLDER = LIST_FOLDER_VOC
# Image format. Working : png, jpg. Other formats might work?
FORMAT = "jpg"
# ??????
SEMANTIC_SEG_FOLDER = SEMANTIC_SEG_FOLDER_VOC
# Output directory (existing) for generated tfrecord files
OUTPUT_DIR="tfrec_output/"

build_voc2012_data.FLAGS.list_folder=LIST_FOLDER
build_voc2012_data.FLAGS.image_folder=IMAGE_FOLDER
build_voc2012_data.FLAGS.image_format=FORMAT # png | jpg
build_voc2012_data.FLAGS.semantic_segmentation_folder=SEMANTIC_SEG_FOLDER
build_voc2012_data.FLAGS.output_dir=OUTPUT_DIR

build_voc2012_data.main(None)
