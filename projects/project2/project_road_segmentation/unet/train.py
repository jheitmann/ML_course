import argparse
import numpy as np
import skimage.io as io
import os

from keras.callbacks import ModelCheckpoint
from model import unet
from preprocessing import extract_data, extract_labels, get_train_generator

TRAINING_PATH = "data/train"
IMG_SUBFOLDER = "image"
GT_SUBFOLDER = "label"
N_TRAIN_IMAGES = 100

TRAIN_IMG_PATH = os.path.join(TRAINING_PATH, IMG_SUBFOLDER)
TRAIN_GT_PATH = os.path.join(TRAINING_PATH, GT_SUBFOLDER)

parser = argparse.ArgumentParser()
parser.add_argument("img_height", type=int, choices=[256, 400],
                    help="image height in pixels")
parser.add_argument("batch_size", type=int, help="training batch size")
parser.add_argument("epochs", type=int, help="number of training epochs")
parser.add_argument("validation_split", type=float, help="ratio of data used to evaluate the model")
parser.add_argument("-rgb", "--rgb_images", help="train with 3 input channels",
                    action="store_true")
parser.add_argument("-aug", "--augmented", help="use augmented dataset",
                    action="store_true")
args = parser.parse_args()

img_height = args.img_height
n_channels = 3 if args.rgb_images else 1
batch_size = args.batch_size
epochs = args.epochs
validation_split = args.validation_split

print(f"Training on images of size {img_height}*{img_height} with {n_channels} input channel(s).")

if (not args.augmented):
    print("Using raw data for training")

    imgs = extract_data(TRAIN_IMG_PATH, N_TRAIN_IMAGES, img_height, args.rgb_images)
    #print(imgs.shape)
    #img0 = imgs[0]
    #print(img0.shape)
    #print(img0[:5,:5])

    gt_imgs = extract_labels(TRAIN_GT_PATH, N_TRAIN_IMAGES, img_height)
    #print(gt_imgs.shape)
    #gt0 = gt_imgs[0]
    #print(gt0.shape)
    #print(gt0[:5,:5])
    #print(np.unique(gt0, return_counts=True))

    input_size = (img_height, img_height, n_channels)
    model = unet(input_size)
    ckpt_file = "results/unet_{}_{}.hdf5".format("rgb" if args.rgb_images else "bw", img_height)
    model_checkpoint = ModelCheckpoint(ckpt_file, monitor='loss', verbose=1, save_best_only=True)
    model.fit(x=imgs, y=gt_imgs, batch_size=batch_size, epochs=epochs, verbose=1,
                  validation_split=validation_split, shuffle=True, callbacks=[model_checkpoint]) # shuffle=False
    
else:
    print("Using augmented dataset")

    imgs = extract_data(TRAIN_IMG_PATH, N_TRAIN_IMAGES, img_height, args.rgb_images, verbose=False)
    
    gt_imgs = extract_labels(TRAIN_GT_PATH, N_TRAIN_IMAGES, img_height)

    input_size = (img_height, img_height, n_channels)
    model = unet(input_size)
    ckpt_file = "results/unet_{}_{}_aug.hdf5".format("rgb" if args.rgb_images else "bw", img_height)
    model_checkpoint = ModelCheckpoint(ckpt_file, monitor='loss', verbose=1, save_best_only=True)
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    save_to_dir = "data/aug"
    steps_per_epoch = 10
    train_generator = get_train_generator(batch_size, TRAINING_PATH, IMG_SUBFOLDER, GT_SUBFOLDER, data_gen_args, save_to_dir=save_to_dir, target_size=(img_height, img_height))
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=[model_checkpoint])
