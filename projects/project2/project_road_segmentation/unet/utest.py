import argparse
import numpy as np

from model import unet
from preprocessing import extract_data
from postprocessing import predictions_to_masks, masks_to_submission
import os

TESTING_PATH = "data/test/image/"
TRAINING_PATH = "data/train/image/"
PREDS_PATH = "results/label/"
RESULTS_PATH = "results/"
N_TEST_IMAGES = 50
N_TRAIN_IMAGES = 100
SUBM_PATH = "results/output.csv"

def main(img_height, rgb, aug, t):
    n_channels = 3 if rgb else 1

    if t:
        imgs = extract_data(TRAINING_PATH, "satImage_", N_TRAIN_IMAGES, img_height, rgb)
    else:
        imgs = extract_data(TESTING_PATH, "test_", N_TEST_IMAGES, img_height, rgb)

    if not aug:
        ckpt_file = os.path.join(RESULTS_PATH, "unet_{}_{}.hdf5".format("rgb" if rgb else "bw", str(img_height)))
    else:
        ckpt_file = os.path.join(RESULTS_PATH, "unet_{}_{}_aug.hdf5".format("rgb" if rgb else "bw", str(img_height)))

    print('ckpt', ckpt_file)
    input_size = (img_height,img_height,n_channels)
    model = unet(input_size, pretrained_weights=ckpt_file)
    print('input_size', input_size)
    preds = model.predict(imgs, batch_size=1, verbose=1)
    print('preds shape', preds.shape)
    print('generating predicted masks in', PREDS_PATH)
    pred_mask_files, logit_mask_files = predictions_to_masks(PREDS_PATH, preds, img_height)
    print('generating submission at', SUBM_PATH)
    masks_to_submission(SUBM_PATH, pred_mask_files, start_from_0=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_height", type=int, choices=[256, 400],
                        help="image height in pixels")
    parser.add_argument("-rgb", "--rgb_images", help="train with 3 input channels",
                        action="store_true")
    parser.add_argument("-aug", "--augmented", help="use augmented dataset",
                        action="store_true")
    parser.add_argument("-t", "--training", help="predict training set instead of testing",
                        action="store_true")                    
    args = parser.parse_args()

    main(args.img_height, args.rgb_images, args.augmented, args.training)
