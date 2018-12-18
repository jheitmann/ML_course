import argparse
import numpy as np
import skimage.io as io
import os
from datetime import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import common
from model import unet
from preprocessing import extract_data, extract_labels, get_generators, split_data
from setup_env import check_env, prepare_train


def main(img_height, batch_size, epochs, steps_per_epoch, rgb=False, aug=False, monitor=None,
        pretrained_weights=None, chosen_validation=False, use_reducelr=True):
    """
    Args:
        img_height: size into which images and masks are resampled, and with which the keras model InputLayer is defined
        batch_size: size of batches used in training
        epochs: number of training epochs
        steps_per_epoch: number of steps of training out of total steps, remaining steps are used for validation 
        rgb: bool set True when using 3 channels for using. Otherwise, channels are averaged into greyscale for training
        aug: bool set True for augmenting the original dataset with transformations such as rotations and flips
        monitor: [acc|loss|val_acc|val_loss] name of metric used for keeping checkpoints. val_* are only usable when steps_per_epoch < |steps|
        pretrained_weights: optional path to a past checkpoint, which is then used as initial weights for training
        use_reducelr: bool set True for using a ReduceLROnPlateau callback, rescaling the learning rate in case of stagnating training performance
    Raises:
        AssertionError: when encountering discrepancies in pretrained_weights/current_model rgb,aug,img_height parameters
    """
    prepare_train(os.getcwd(), verbose=True)

    if pretrained_weights:
        assert str(img_height) in pretrained_weights, "Wrong img_height pretrained weights"
        assert ("rgb" if rgb else "bw") in pretrained_weights, "Wrong color mode pretrained weights"
        assert ("aug" in pretrained_weights) if aug else True, "aug bool not matching pretrained weights"
    n_channels = 3 if rgb else 1
    validation_split = (100 - steps_per_epoch) / 100.0

    print(f"Training on images of size {img_height}*{img_height} with {n_channels} input channel(s).")

    input_size = (img_height, img_height, n_channels)
    model = unet(input_size, pretrained_weights=pretrained_weights)
    
    if chosen_validation:
        val_imgs = extract_data(common.SPLIT_VAL_IMG_PATH, common.N_SPLIT_VAL, img_height, rgb)
        val_gt_imgs = extract_labels(common.SPLIT_VAL_GT_PATH, common.N_SPLIT_VAL, img_height)
        validation_data = (val_imgs, val_gt_imgs)
    else:
        validation_data = None

    if (not aug):
        print("Using raw data for training")

        if chosen_validation:
            imgs = extract_data(common.SPLIT_TRAIN_IMG_PATH, common.N_SPLIT_TRAIN, img_height, rgb)
            gt_imgs = extract_labels(common.SPLIT_TRAIN_GT_PATH, common.N_SPLIT_TRAIN, img_height)
        else:
            imgs = extract_data(common.TRAIN_IMG_PATH, common.N_TRAIN_IMAGES, img_height, rgb)
            gt_imgs = extract_labels(common.TRAIN_GT_PATH, common.N_TRAIN_IMAGES, img_height)

        monitor = "acc" if not monitor else monitor
        if use_reducelr:
            print(f"Using ReduceLROnPlateau on {monitor} w. factor {0.5},patience {5}, min_lr {0.001}")
            reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, min_lr=0.001)
        hdf5_name = "unet_{}_{}_{}.hdf5".format("rgb" if rgb else "bw", img_height, str(datetime.now()).replace(':', '_').replace(' ', '_'))
        print("hdf5 name:", hdf5_name)
        ckpt_file = os.path.join(common.RESULTS_PATH, hdf5_name)
        model_checkpoint = ModelCheckpoint(ckpt_file, monitor=monitor, verbose=1, save_best_only=True)
        model.fit(x=imgs, y=gt_imgs, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split, 
                    validation_data=validation_data, shuffle=False, callbacks=[model_checkpoint, reduce_lr])
        
    else:
        print("Using augmented dataset")

        hdf5_name = "unet_{}_{}_{}_aug.hdf5".format("rgb" if rgb else "bw", img_height, str(datetime.now()).replace(':', '_').replace(' ', '_'))
        print("hdf5 name:", hdf5_name)
        ckpt_file = os.path.join(common.RESULTS_PATH, hdf5_name)
        data_gen_args = dict(rotation_range=90, fill_mode='reflect', horizontal_flip=True, vertical_flip=True) # shear_range = 0.01, zoom_range = 0.2

        if 0 < validation_split < 1 and not chosen_validation: #Only add validation_split if in (0;1) cf keras doc, to allow debugging with 100 steps (validation_split of 0 is not accepted)
            data_gen_args["validation_split"] = validation_split

        monitor = "val_acc" if not monitor else monitor
        if use_reducelr:
            print(f"Using ReduceLROnPlateau on {monitor} w. factor {0.5},patience {5}, min_lr {0.001}")
            reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, min_lr=0.001)
        print("Monitoring with", monitor)
        """
        if "val" in monitor:
            assert "validation_split" in data_gen_args, "Monitoring a val metric with invalid validation_split"
        """

        model_checkpoint = ModelCheckpoint(ckpt_file, monitor=monitor, verbose=1, save_best_only=True)
        color_mode = "rgb" if rgb else "grayscale"

        if chosen_validation:
            train_generator, _ = get_generators(batch_size, common.SPLIT_TRAIN_PATH, common.IMG_SUBFOLDER, common.GT_SUBFOLDER, 
                                                                data_gen_args,  target_size=(img_height,img_height), color_mode=color_mode) # save_to_dir=AUG_SAVE_PATH
        else: 
            train_generator, validation_data = get_generators(batch_size, common.TRAIN_PATH, common.IMG_SUBFOLDER, common.GT_SUBFOLDER, 
                                                                data_gen_args,  target_size=(img_height,img_height), color_mode=color_mode) # save_to_dir=AUG_SAVE_PATH
        # Create validation parameters dict. passed to fit_generator(.) if using validation split in (0;1) else create an empty parameter dict
        validation_params = dict(validation_data=validation_data, validation_steps=(common.N_TRAIN_IMAGES - steps_per_epoch)) if "validation_split" in data_gen_args or chosen_validation else {}
        model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=[model_checkpoint], **validation_params) #, reduce_lr
    
    return ckpt_file

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_height", type=int,
                        help="image height in pixels")
    parser.add_argument("batch_size", type=int, help="training batch size")
    parser.add_argument("epochs", type=int, help="number of training epochs")
    parser.add_argument("steps_per_epoch", type=float, choices=range(1, common.N_TRAIN_IMAGES+1),
                        help="number of training images per epoch")
    parser.add_argument("-monitor", "--monitor", type=str, choices=["acc", "loss", "val_acc", "val_loss"],
                        default="", help="monitor metric for checkpoint")
    parser.add_argument("-rgb", "--rgb_images", help="train with 3 input channels",
                        action="store_true")
    parser.add_argument("-aug", "--augmented", help="use augmented dataset",
                        action="store_true")
    parser.add_argument("-pre", "--preweights", type=str, help="path to pretrained weights")
    args = parser.parse_args()

    main(args.img_height, args.batch_size, args.epochs, args.steps_per_epoch, args.rgb_images, args.augmented, args.monitor, args.preweights)
