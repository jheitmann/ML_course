import argparse
import numpy as np
import os
from datetime import datetime

import common
from model import unet
from preprocessing import extract_data, extract_labels, get_checkpoint, get_generators
from setup_env import check_env, prepare_train


def main(img_height, batch_size, epochs, steps_per_epoch, aug, chosen_validation, rgb,
    pretrained_weights, monitor, *, data_gen_args=common.DEFAULT_GEN_ARGS, root_folder=None):
    """
    Args:
        img_height: size into which images and masks are resampled, and with which the keras model InputLayer is defined
        batch_size: size of batches used in training
        epochs: number of training epochs
        steps_per_epoch: number of steps of training out of total steps, remaining steps are used for validation 
        aug: bool set True for augmenting the original dataset with transformations such as rotations and flips
        chosen_validation: bool set True when model is validated on a chosen dataset, instead of a random one
        rgb: bool set True when using 3 channels. Otherwise, channels are averaged into greyscale for training
        pretrained_weights: optional path to a past checkpoint, which is then used as initial weights for training
        monitor: [acc|loss|val_acc|val_loss] name of metric used for keeping checkpoints. val_* are only usable when steps_per_epoch < |steps|
        root_folder: use to override root_folder=os.getcwd (Typically when using main() in Google Colab)
        data_gen_args: args passed to each ImageDataGenerator used during training.
    Raises:
        AssertionError: when encountering discrepancies in pretrained_weights/current_model rgb, img_height parameters
    """
    prepare_train(os.getcwd() if not root_folder else root_folder, verbose=True)

    n_channels = 3 if rgb else 1
    validation_split = (common.N_TRAIN_IMAGES - steps_per_epoch) / float(common.N_TRAIN_IMAGES)
    input_size = (img_height,img_height,n_channels)
    model = unet(input_size, pretrained_weights=pretrained_weights)

    # Create validation parameters dict. passed to fit(.)/fit_generator(.)
    validation_params = {"validation_data": None}

    if chosen_validation:
        val_imgs = extract_data(common.SPLIT_VAL_IMG_PATH, common.N_SPLIT_VAL, img_height, rgb)
        val_gt_imgs = extract_labels(common.SPLIT_VAL_GT_PATH, common.N_SPLIT_VAL, img_height)
        validation_data = (val_imgs, val_gt_imgs)
        validation_params["validation_data"] = validation_data
        monitor = monitor if monitor else "val_acc"
    else:
        validation_steps = common.N_TRAIN_IMAGES - steps_per_epoch
        if validation_split > 0:
            validation_params['validation_steps'] = validation_steps
        
        if monitor and "val" in monitor:
            assert validation_split > 0, "Monitoring a val metric with invalid validation_split"
        else:
            monitor = monitor if monitor else ("val_acc" if validation_split > 0 else "acc")

    if pretrained_weights:
        assert str(img_height) in pretrained_weights, "Wrong img_height pretrained weights"
        assert ("rgb" if rgb else "bw") in pretrained_weights, "Wrong color mode pretrained weights"
    
    print(f"Training on images of size {img_height}*{img_height} with {n_channels} input channel(s).")
    print("Using {} dataset {} chosen validation for training".format("raw" if not aug else "augmented", "with" if chosen_validation else "without"))
    print("Monitoring with", monitor)

    if not aug:
        if chosen_validation:
            imgs = extract_data(common.SPLIT_TRAIN_IMG_PATH, common.N_SPLIT_TRAIN, img_height, rgb)
            gt_imgs = extract_labels(common.SPLIT_TRAIN_GT_PATH, common.N_SPLIT_TRAIN, img_height)
        else:
            imgs = extract_data(common.TRAIN_IMG_PATH, common.N_TRAIN_IMAGES, img_height, rgb)
            gt_imgs = extract_labels(common.TRAIN_GT_PATH, common.N_TRAIN_IMAGES, img_height)

        ckpt_file, model_checkpoint = get_checkpoint(img_height, rgb, monitor)
        model.fit(x=imgs, y=gt_imgs, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split, 
                    validation_data=validation_params['validation_data'], shuffle=False, callbacks=[model_checkpoint])
        
    else:
        color_mode = "rgb" if rgb else "grayscale"

        # save_to_dir=common.AUG_SAVE_PATH
        if chosen_validation:
            train_generator, _ = get_generators(batch_size, common.SPLIT_TRAIN_PATH, common.IMG_SUBFOLDER, common.GT_SUBFOLDER, 
                                                    data_gen_args,  target_size=(img_height,img_height), color_mode=color_mode)   
        else:
            data_gen_args["validation_split"] = validation_split 
            train_generator, validation_generator = get_generators(batch_size, common.TRAIN_PATH, common.IMG_SUBFOLDER, common.GT_SUBFOLDER, 
                                                                    data_gen_args,  target_size=(img_height,img_height), color_mode=color_mode)
            validation_params["validation_data"] = validation_generator 
        
        ckpt_file, model_checkpoint = get_checkpoint(img_height, rgb, monitor)
        model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=[model_checkpoint], **validation_params)
    
    return ckpt_file

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_height", type=int,
                        help="image height in pixels")
    parser.add_argument("batch_size", type=int, help="training batch size")
    parser.add_argument("epochs", type=int, help="number of training epochs")
    parser.add_argument("steps_per_epoch", type=float, choices=range(1, common.N_TRAIN_IMAGES+1),
                        help="number of training images per epoch")
    parser.add_argument("-aug", "--augmented", help="use augmented dataset",
                        action="store_true")
    parser.add_argument("-cv", "--chosen_validation", help="use a chosen validation set",
                        action="store_true")
    parser.add_argument("-rgb", "--rgb_images", help="train with 3 input channels",
                        action="store_true")                        
    parser.add_argument("-pre", "--preweights", type=str, help="path to pretrained weights")
    parser.add_argument("--monitor", type=str, choices=["acc", "loss", "val_acc", "val_loss"],
                        default="", help="monitor metric for checkpoint")
    parser.add_argument("--rot", type=float,
                        help="rotation augmentation for ImageDataGenerator (default:90)")
    paser.add_argument("--zoom", type=float,
                        help="zoom augmentation for ImageDataGenerator (default:None)")
    parser.add_argument("--shift", type=float,
                        help="shift (x and y) augmentation for ImageDataGenerator (default:None)")
    args = parser.parse_args()
    print(args.rot, args.shift, args.zoom)
    data_gen_args = dict(
        rotation_range=args.rot if args.rot else common.DEFAULT_GEN_ARGS.rotation,
        width_shift_range=args.shift if args.shift else 0,
        height_shift_range=args.shift if args.shift else 0,
        zoom_range=args.zoom if args.zoom else 0  
    )
    main(args.img_height, args.batch_size, args.epochs, args.steps_per_epoch, args.augmented,
        args.chosen_validation, args.rgb_images, args.preweights, args.monitor,
        data_gen_args=data_gen_args)
