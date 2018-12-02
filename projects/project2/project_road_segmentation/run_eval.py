
import sys
sys.path.append("./models/research/")
sys.path.append("./models/research/slim/")
from deeplab import eval

DATASET_DIR_PROJ = "eval_dataset/"

eval.FLAGS.eval_split="val"
eval.FLAGS.model_variant="xception_65"
eval.FLAGS.atrous_rates=[6, 12, 18]
eval.FLAGS.output_stride=16
eval.FLAGS.decoder_output_stride=4
eval.FLAGS.eval_crop_size=[513, 513]
eval.FLAGS.eval_batch_size=1
eval.FLAGS.dataset="pascal_voc_seg"
eval.FLAGS.checkpoint_dir="train_output/"
eval.FLAGS.eval_logdir="eval_output/"
eval.FLAGS.dataset_dir=DATASET_DIR_PROJ

eval.main(None)