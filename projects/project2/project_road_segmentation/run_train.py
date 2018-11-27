
import sys
sys.path.append("./models/research/")
sys.path.append("./models/research/slim/")
from deeplab import train

TF_INITIAL_CKPT = "initial_ckpts/deeplabv3_pascal_train_aug/model.ckpt"
TRAIN_LOGDIR_PROJ = "datasets/proj_dataset/training/train_outputs/"
DATASET_DIR_PROJ = "datasets/proj_dataset/training/tfrecord/"
TRAIN_LOGDIR_VOC = "datasets/pascal_voc_seg/exp/train_on_train_set/train/"
DATASET_DIR_VOC = "datasets/pascal_voc_seg/tfrecord/"

train.FLAGS.training_number_of_steps=30000
train.FLAGS.train_split="train"
train.FLAGS.model_variant="xception_65"
train.FLAGS.atrous_rates=[6, 12, 18]
train.FLAGS.output_stride=16
train.FLAGS.decoder_output_stride=4
train.FLAGS.train_crop_size=[513, 513]
train.FLAGS.train_batch_size=1
train.FLAGS.dataset="pascal_voc_seg"
train.FLAGS.tf_initial_checkpoint=TF_INITIAL_CKPT
train.FLAGS.train_logdir=TRAIN_LOGDIR_PROJ
train.FLAGS.dataset_dir=DATASET_DIR_PROJ

train.main(None)

"""
python deeplab/train.py \
    --logtostderr \
    --train_batch_size=1 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
"""