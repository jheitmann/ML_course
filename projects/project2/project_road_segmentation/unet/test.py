import argparse
import numpy as np

from model import unet
from preprocessing import extract_data
from postprocessing import predictions_to_masks


TESTING_PATH = "data/test/image/"
PREDS_PATH = "results/label/"
N_TEST_IMAGES = 50

parser = argparse.ArgumentParser()
parser.add_argument("img_height", type=int, choices=[256, 400],
                    help="image height in pixels")
parser.add_argument("-rgb", "--rgb_images", help="train with 3 input channels",
                    action="store_true")
args = parser.parse_args()

img_height = args.img_height
n_channels = 3 if args.rgb_images else 1

imgs = extract_data(TESTING_PATH + "test", N_TEST_IMAGES, img_height, args.rgb_images)

ckpt_file = "results/unet_{}_{}.hdf5".format("rgb" if args.rgb_images else "bw", str(img_height))
input_size = (img_height,img_height,n_channels)
model = unet(input_size, pretrained_weights=ckpt_file)

preds = model.predict(imgs, batch_size=1, verbose=1)
print(preds.shape)
predictions_to_masks(PREDS_PATH, preds, img_height)
