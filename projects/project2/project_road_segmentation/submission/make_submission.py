
import sys
sys.path.append('..')
import os
from PIL import Image
from util import resize

RESIZED_DIR = "resized/"
PIXEL_THRES_DIR = "pixel_thres/"
PATCH_THRES_DIR = "patch_thres/"

TEST_RESULTS_PATH = "../unet/data/roadseg/test_res_4"

def map_pixels(im, f, mode='RGB', size=(400, 400)):
    im2 = Image.new(mode, size)
    for pi in range(im.size[0]):
        for pj in range(im.size[1]):
            im2.putpixel((pi, pj), f(pi, pj, im.getpixel((pi, pj))))
    return im2

def main(images_folder):
    resize.resize(images_folder, RESIZED_DIR, (400, 400))

    for fname in os.listdir(RESIZED_DIR):
        fpath = os.path.join(RESIZED_DIR, fname)

        os.mkdir(PIXEL_THRES_DIR)
        im = Image.open(fpath)
        im_pixel_thres = map_pixels(im, pixel_thres) #lambda x, y, c: 1 if c > .5 else 0)
        im_pixel_thres.save(os.path.join(PIXEL_THRES_DIR, fname))

        os.mkdir(PATCH_THRES_DIR)
        patches = []
        im_patches_thres = map_pixels(im_pixel_thres, lambda x, y, c: 1 if patches[x / 16, y / 16] else 0)
        im_patches_thres.save(os.path.join(PATCH_THRES_DIR, fname))

def pixel_thres(x, y, c):
    return 255 if c > .5 else 0

if __name__=="__main__":
    for d in (RESIZED_DIR, PIXEL_THRES_DIR, PATCH_THRES_DIR):
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        os.rmdir(d)
    main(TEST_RESULTS_PATH)
