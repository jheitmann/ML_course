
import os
import sys
import random
import math
import tensorflow as tf
from deeplab.datasets import build_data

class PARAM:
    output_dir = "tfrec_output"
    train_image_folder = "datasets/roadseg/images" # Training dataset images root
    train_image_label_folder = "datasets/roadseg/groundtruth/" # and their masks
    val_image_folder = "datasets/roadseg/images"  # For later, need to also use validation/testing images
    val_image_label_folder = "datasets/roadseg/groundtruth/"  # and their masks
    num_shards = 1 # Number of shards of dataset (mostly useful for distributed training, we don't need more than 1)
    image_nchannels = 3 # Channels in image : 3 for (R, G, B), 4 if alpha channel
    label_nchannels = 1  # Channels in masks (works with 1 but not sure if really is 1)
    image_ext = 'png' # Only jpeg/jpg or png
    label_ext = 'png' # Only jpeg/jpg or png
    
def _convert_dataset(tfrec_name, dataset_dir, dataset_label_dir):
    img_names = tf.gfile.Glob(os.path.join(dataset_dir, f'*.{PARAM.image_ext}'))
    random.shuffle(img_names)
    seg_names = []

    for f in img_names:
        basename = os.path.basename(f).split('.')[0]
        seg = os.path.join(dataset_label_dir, f'{basename}.{PARAM.label_ext}')
        seg_names.append(seg)

    num_images = len(img_names)
    num_per_shard = int(math.ceil(num_images / float(PARAM.num_shards)))

    image_reader = build_data.ImageReader('png' if PARAM.image_ext == 'png' else 'jpeg', channels=PARAM.image_nchannels)
    label_reader = build_data.ImageReader('png' if PARAM.image_ext == 'png' else 'jpeg', channels=PARAM.label_nchannels)

    for shard_id in range(PARAM.num_shards):
        output_filename = os.path.join(PARAM.output_dir, '%s-%05d-of-%05d.tfrecord' % (tfrec_name, shard_id, PARAM.num_shards))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                print(f'\r>> Converting image {i+1}/{num_images} shard {shard_id}')
                # Read the image.
                image_filename = img_names[i]
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_filename = seg_names[i]
                seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(image_data, img_names[i], height, width, seg_data)                
                tfrecord_writer.write(example.SerializeToString())

def main():
    tf.gfile.MakeDirs(PARAM.output_dir)
    _convert_dataset('train', PARAM.train_image_folder, PARAM.train_image_label_folder)
    #_convert_dataset('val', FLAGS.val_image_folder, FLAGS.val_image_label_folder)

if __name__ == '__main__':
    main()
