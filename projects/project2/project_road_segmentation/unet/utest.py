import argparse
import numpy as np
import os

from common import TEST_IMG_PATH, TRAIN_IMG_PATH, TESTING_PATH_FOURSPLIT, RESULTS_PATH, SUBM_PATH,\
                    N_TEST_IMAGES, N_TRAIN_IMAGES, TRAIN_IMG_HEIGHT, TEST_IMG_HEIGHT
from model import unet
from postprocessing import four_split_mean, predictions_to_masks, masks_to_submission
from preprocessing import extract_data
from setup_env import check_env, prepare_test, prepare_train, gen_four_split


REGENERATE_FOUR_SPLIT = False


def main(ckpt_path, t, four_split, foreground_threshold=0.25): # change to p_threshold
    #prepare_test(os.getcwd(), verbose=True)

    rgb = "rgb" in ckpt_path
    n_channels = 3 if rgb else 1
    # Extract the image height from the checkpoint file name (thus need to be named from our script)
    img_height = int(os.path.basename(ckpt_path).split("_")[2].split('.')[0])

    if four_split:
        assert not t, "Four split on training dataset is a bad idea (images already at correct scale)"
        if REGENERATE_FOUR_SPLIT:
            gen_four_split(TEST_IMG_PATH, TESTING_PATH_FOURSPLIT)
        imgs = extract_data(TESTING_PATH_FOURSPLIT, N_TEST_IMAGES * 4, img_height, rgb)
    else:
        if t:
            imgs = extract_data(TRAIN_IMG_PATH, N_TRAIN_IMAGES, img_height, rgb)
        else:
            imgs = extract_data(TEST_IMG_PATH, N_TEST_IMAGES, img_height, rgb)
    
    print('ckpt', ckpt_path) # ckpt_file    
    
    input_size = (img_height,img_height,n_channels)
    print('Neural network input size:', input_size)
    model = unet(input_size, pretrained_weights=ckpt_path)
    
    preds = model.predict(imgs, batch_size=1, verbose=1)
    print('Predictions shape:', preds.shape)
    
    print('Generating predicted masks in', RESULTS_PATH)
    result_path = RESULTS_PATH + ("train/" if t else "test/")
    test_name = os.path.join(TRAIN_IMG_PATH, "satImage") if t else os.path.join(TEST_IMG_PATH, "test")
    output_height = TRAIN_IMG_HEIGHT if t or four_split else TEST_IMG_HEIGHT
    predicted_mask_files = predictions_to_masks(result_path, test_name, preds, output_height, 
                                                    four_split, TEST_IMG_HEIGHT, use_mean=True) # False
    
    print('Generating submission at', SUBM_PATH)
    masks_to_submission(SUBM_PATH, predicted_mask_files, foreground_threshold=foreground_threshold)

    return SUBM_PATH

if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str,
                        help="path to ckpt file")
    parser.add_argument("-t", "--training", help="predict training set instead of testing",
                        action="store_true")
    parser.add_argument("-fs", "--four_split", help="take mean of four predictions on testing images",
                        action="store_true")        
    args = parser.parse_args()
    main(args.ckpt_path, args.training, args.four_split)
