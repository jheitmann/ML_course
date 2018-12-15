import argparse
import numpy as np

from model import unet
from preprocessing import extract_data
from postprocessing import predictions_to_masks, masks_to_submission
import os

TESTING_PATH = "data/test/image/"
TRAINING_PATH = "data/train/image/"
RESULT_PATH = "results/"
N_TEST_IMAGES = 50
N_TRAIN_IMAGES = 100
SUBM_PATH = "results/output.csv"

#def main(img_height, rgb, aug, t):
def main(ckpt_path, t, foreground_threshold=0.25):
    rgb = "rgb" in ckpt_path
    n_channels = 3 if rgb else 1
    img_height = int(os.path.basename(ckpt_path).split("_")[2].split('.')[0])
    aug = "aug" in ckpt_path

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
    print('ckpt', ckpt_path)#ckpt_file    
    input_size = (img_height,img_height,n_channels)
    model = unet(input_size, pretrained_weights=ckpt_path)
    print('input_size', input_size)
    preds = model.predict(imgs, batch_size=1, verbose=1)
    print('preds shape', preds.shape)
    print('generating predicted masks in', RESULT_PATH)
    test_name = TRAINING_PATH + "satImage" if t else TESTING_PATH + "test"
    predicted_mask_files = predictions_to_masks(RESULT_PATH, test_name, preds)
    print('generating submission at', SUBM_PATH)
    masks_to_submission(SUBM_PATH, predicted_mask_files,foreground_threshold=foreground_threshold)

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
         
    args = parser.parse_args()
    # main(args.img_height, args.rgb_images, args.augmented, args.training)
    main(args.ckpt_path, args.training)
