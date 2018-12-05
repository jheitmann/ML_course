
import sys
sys.path.append('..')
import os
from PIL import Image
from util import resize

RESIZED_DIR = "resized/"
PIXEL_THRES_DIR = "pixel_thres/"
PATCH_THRES_DIR = "patch_thres/"

TEST_RESULTS_PATH = "../unet/data/roadseg/test_res_4"
2

def main(images_folder):
    resize.resize(images_folder, RESIZED_DIR, (400, 400))

    os.mkdir(PIXEL_THRES_DIR)  
    os.mkdir(PATCH_THRES_DIR)  
    for fname in os.listdir(RESIZED_DIR):
        fpath = os.path.join(RESIZED_DIR, fname)
    
        im = Image.open(fpath)
        print('opening', fpath, 'mode', im.mode)
        im = im.convert('L')

        im2 = Image.new('L', (400, 400))
        for pi in range(im.size[0]):
            for pj in range(im.size[1]):
                #im2.putpixel((pi, pj), 255 if im.getpixel((pi, pj)) > 127 else 0)
                im2.putpixel((pi, pj), im.getpixel((pi, pj)))
        im2.save(os.path.join(PIXEL_THRES_DIR, fname))

        #im_pixel_thres = map_pixels(im, pixel_thres) #lambda x, y, c: 1 if c > .5 else 0)

        patches = []
        #im_patches_thres = map_pixels(im_pixel_thres, lambda x, y, c: 1 if patches[x / 16, y / 16] else 0)
        #im_patches_thres.save(os.path.join(PATCH_THRES_DIR, fname))

if __name__=="__main__":
    for d in (RESIZED_DIR, PIXEL_THRES_DIR, PATCH_THRES_DIR):
        print("query", d)
        if not os.path.isdir(d):
            print("does not exists. continue")
            continue
        for f in os.listdir(d):
            print("rm", f, "in", d)
            os.remove(os.path.join(d, f))
        print("rm", d)
        os.rmdir(d)
    main(TEST_RESULTS_PATH)
