import argparse
import numpy as np

from model import unet
from preprocessing import extract_data
from postprocessing import four_split_mean, predictions_to_masks, masks_to_submission, gen_four_split
import os

TESTING_PATH = "data/test/image/"
TRAINING_PATH = "data/train/image/"
TESTING_PATH_FOURSPLIT = "data/test/foursplit/"
RESULT_PATH = "results/"
N_TEST_IMAGES = 50
N_TRAIN_IMAGES = 100
TEST_IMG_HEIGHT = 608
TRAIN_IMG_HEIGHT = 400
SUBM_PATH = "results/output.csv"

REGENERATE_FOUR_SPLIT = False

#def main(img_height, rgb, aug, t):
def main(ckpt_path, t, four_split, foreground_threshold=0.25): # change to p_threshold

    rgb = "rgb" in ckpt_path
    n_channels = 3 if rgb else 1
    img_height = int(os.path.basename(ckpt_path).split("_")[2].split('.')[0])

    if four_split:
        assert not t, "Four split on training dataset is a bad idea (images already at correct scale)"
        if REGENERATE_FOUR_SPLIT:
            gen_four_split(TESTING_PATH, TESTING_PATH_FOURSPLIT)
        imgs = extract_data(TESTING_PATH_FOURSPLIT, "test_", N_TEST_IMAGES * 4, img_height, rgb)
    else:
        if t:
            imgs = extract_data(TRAINING_PATH, "satImage_", N_TRAIN_IMAGES, img_height, rgb)
        else:
            imgs = extract_data(TESTING_PATH, "test_", N_TEST_IMAGES, img_height, rgb)
    """
    if not aug:
        ckpt_file = os.path.join(RESULT_PATH, "unet_{}_{}.hdf5".format("rgb" if rgb else "bw", str(img_height)))
    else:
        ckpt_file = os.path.join(RESULT_PATH, "unet_{}_{}_aug.hdf5".format("rgb" if rgb else "bw", str(img_height)))
    """
    
    print('ckpt', ckpt_path) # ckpt_file    
    
    input_size = (img_height,img_height,n_channels)
    print('Neural network input size:', input_size)
    model = unet(input_size, pretrained_weights=ckpt_path)
    
    preds = model.predict(imgs, batch_size=1, verbose=1)
    print('Predictions shape:', preds.shape)
    
    print('Generating predicted masks in', RESULT_PATH)
    result_path = RESULT_PATH + ("train/" if t else "test/")
    test_name = TRAINING_PATH + "satImage" if t else TESTING_PATH + "test"
    output_height = TRAIN_IMG_HEIGHT if t or four_split else TEST_IMG_HEIGHT
    predicted_mask_files = predictions_to_masks(result_path, test_name, preds, output_height, four_split, TEST_IMG_HEIGHT)
    
    print('Generating submission at', SUBM_PATH)
    masks_to_submission(SUBM_PATH, predicted_mask_files, foreground_threshold=foreground_threshold)

if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    """
    parser.add_argument("img_height", type=int,
                        help="image height in pixels")
    parser.add_argument("-rgb", "--rgb_images", help="train with 3 input channels",
                        action="store_true")
    parser.add_argument("-aug", "--augmented", help="use augmented dataset",
                        action="store_true")
    parser.add_argument("-t", "--training", help="predict training set instead of testing",
                        action="store_true")                    
    args = parser.parse_args()
    """
    parser.add_argument("ckpt_path", type=str,
                        help="path to ckpt file")
    parser.add_argument("-t", "--training", help="predict training set instead of testing",
                        action="store_true")
    parser.add_argument("-fs", "--four_split", help="take mean of four predictions on testing images",
                        action="store_true")        
         
    args = parser.parse_args()
    # main(args.img_height, args.rgb_images, args.augmented, args.training)
    main(args.ckpt_path, args.training, args.four_split)
