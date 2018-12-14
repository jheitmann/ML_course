import argparse
import numpy as np
import skimage.io as io
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model import unet
from preprocessing import extract_data, extract_labels, get_generators, split_data

TRAINING_PATH = "data/train"
IMG_SUBFOLDER = "image"
GT_SUBFOLDER = "label"
N_TRAIN_IMAGES = 100

TRAIN_IMG_PATH = os.path.join(TRAINING_PATH, IMG_SUBFOLDER)
TRAIN_GT_PATH = os.path.join(TRAINING_PATH, GT_SUBFOLDER)

def main(img_height, batch_size, epochs, steps_per_epoch, rgb=False, aug=False, monitor='loss'):
    img_height = img_height
    n_channels = 3 if rgb else 1
    validation_split = (100 - steps_per_epoch) / 100.0

    print(f"Training on images of size {img_height}*{img_height} with {n_channels} input channel(s).")

    if (not aug):
        print("Using raw data for training")

        imgs = extract_data(TRAIN_IMG_PATH, "satImage_", N_TRAIN_IMAGES, img_height, rgb)

        gt_imgs = extract_labels(TRAIN_GT_PATH, N_TRAIN_IMAGES, img_height)

        input_size = (img_height, img_height, n_channels)
        model = unet(input_size)
        ckpt_file = "results/unet_{}_{}.hdf5".format("rgb" if rgb else "bw", img_height)
        model_checkpoint = ModelCheckpoint(ckpt_file, monitor='val_loss', verbose=1, save_best_only=True)
        model.fit(x=imgs, y=gt_imgs, batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_split=validation_split, shuffle=True, callbacks=[model_checkpoint]) # shuffle=False
        
    else:
        print("Using augmented dataset")

        input_size = (img_height, img_height, n_channels)
        model = unet(input_size)
        ckpt_file = "results/unet_{}_{}_aug.hdf5".format("rgb" if rgb else "bw", img_height)

        data_gen_args = dict(rotation_range=90, fill_mode='reflect', horizontal_flip=True, vertical_flip=True) # shear_range = 0.01, zoom_range = 0.2

        if 0 < validation_split < 1: #Only add validation_split if in (0;1) cf keras doc, to allow debugging with 100 steps (validation_split of 0 is not accepted)
            data_gen_args["validation_split"] = validation_split

        monitor = "val_acc" if not monitor else monitor
        print("Monitoring with", monitor)
        if "val" in monitor:
            assert "validation_split" in data_gen_args

        model_checkpoint = ModelCheckpoint(ckpt_file, monitor=monitor, verbose=1, save_best_only=True)
        save_to_dir = "data/train/aug/"
        color_mode = "rgb" if rgb else "grayscale"
        train_generator, validation_generator = get_generators(batch_size, TRAINING_PATH, IMG_SUBFOLDER, GT_SUBFOLDER, data_gen_args,  target_size=(img_height,img_height), color_mode=color_mode, save_to_dir=save_to_dir)
        # Create validation parameters dict. passed to fit_generator(.) if using validation split in (0;1) else create an empty parameter dict
        validation_params = dict(validation_data=validation_generator, validation_steps=(N_TRAIN_IMAGES - steps_per_epoch)) if "validation_split" in data_gen_args else {}
        model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=[model_checkpoint], **validation_params)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_height", type=int, choices=[256, 400],
                        help="image height in pixels")
    parser.add_argument("batch_size", type=int, help="training batch size")
    parser.add_argument("epochs", type=int, help="number of training epochs")
    parser.add_argument("steps_per_epoch", type=float, choices=range(1, N_TRAIN_IMAGES+1),
                        help="number of training images per epoch")
    parser.add_argument("-monitor", "--monitor", type=str, choices=["", "acc", "loss", "val_acc", "val_loss"],
                        default="", help="monitor metric for checkpoint")
    parser.add_argument("-rgb", "--rgb_images", help="train with 3 input channels",
                        action="store_true")
    parser.add_argument("-aug", "--augmented", help="use augmented dataset",
                        action="store_true")
    args = parser.parse_args()

    main(args.img_height, args.batch_size, args.epochs, args.steps_per_epoch, args.rgb_images, args.augmented)
