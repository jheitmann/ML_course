
import sys
sys.path.append('..')
import os
from PIL import Image

import scipy.misc

RESIZED_DIR = "resized/"
PIXEL_THRES_DIR = "pixel_thres/"
PATCH_THRES_DIR = "patch_thres/"

TEST_RESULTS_PATH = "../unet/data/roadseg/test_res_4"


def resize_one(im):
    imr = im.resize((608, 608), Image.ANTIALIAS)
    return imr

def pixel_thres_one(im):
    imr = Image.new('L', im.size)
    for px in range(im.size[0]):
        for py in range(im.size[1]):
            imr.putpixel((px, py), 255 if im.getpixel((px, py)) > 127 else 0)
    return imr

def patch_thres_one(im, patchsize=(16, 16), thres=0.25):
    imr = Image.new('L', im.size)
    ptw, pth = (int(im.size[i] / patchsize[i]) for i in (0, 1))
    for ptx in range(ptw):
        for pty in range(pth):
            ptsum = 0
            for pxx in range(patchsize[0]):
                for pxy in range(patchsize[1]):
                    ptsum += im.getpixel((ptx * patchsize[0] + pxx, pty * patchsize[1] + pxy))
            above_thres = ptsum > (255 * patchsize[0] * patchsize[1]) * thres
            for pxx in range(patchsize[0]):
                for pxy in range(patchsize[1]):
                    imr.putpixel((ptx * patchsize[0] + pxx, pty * patchsize[1] + pxy), 255 if above_thres else 0)
    return imr

def convert_to_L(im):
    """ Converts 16-bit greyscale I mode to 8-bit greyscale L mode """
    imr = Image.new('L', im.size)
    for px in range(im.size[0]):
        for py in range(im.size[1]):            
            rescale=255./65535.
            imr.putpixel((px, py), int(im.getpixel((px, py)) * rescale))
    return imr

def img_to_line(image_fpath):
    """Reads a single image and outputs the strings that should go into the submission file"""
    #img_number = int(re.search(r"\d+", image_filename).group(0))
    img_number = int(image_fpath.split("test_")[1].split("_resized")[0])
    print(img_number)
    im = Image.open(image_fpath)
    patch_size = 16
    for j in range(0, im.size[1], patch_size):
        for i in range(0, im.size[0], patch_size):
            patch = im.getpixel((i, j))
            label = 1 if patch > 0 else 0
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

def write_csv(csv_fpath, img_dir):
    """Converts images into a submission file"""
    with open(csv_fpath, 'w') as f:
        f.write('id,prediction\n')
        for img in os.listdir(img_dir):
            fpath = os.path.join(img_dir, img)
            f.writelines(f'{line}\n' for line in img_to_line(fpath))

def main(images_folder):
    #resize.resize(images_folder, RESIZED_DIR, (400, 400))

    os.mkdir(RESIZED_DIR)
    os.mkdir(PIXEL_THRES_DIR)  
    os.mkdir(PATCH_THRES_DIR)  
    for fname in os.listdir(images_folder):
        fpath = os.path.join(images_folder, fname)
    
        im = Image.open(fpath)
        print('loaded', fpath, 'mode', im.mode)
        im = convert_to_L(im)

        im_rs = resize_one(im)
        im_rs.save(os.path.join(RESIZED_DIR, fname))
        print('resized', os.path.join(RESIZED_DIR, fname), 'mode', im_rs.mode)

        im_pxt = pixel_thres_one(im_rs)
        im_pxt.save(os.path.join(PIXEL_THRES_DIR, fname))
        print('pixel thresholded', os.path.join(PIXEL_THRES_DIR, fname), 'mode', im_pxt.mode)

        im_ptt = patch_thres_one(im_pxt)
        im_ptt.save(os.path.join(PATCH_THRES_DIR, fname))
        print('patch thresholded', os.path.join(PATCH_THRES_DIR, fname), 'mode', im_ptt.mode)

    write_csv("output.csv", "patch_thres")

def clean():
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

if __name__=="__main__":
    clean()
    main(TEST_RESULTS_PATH)
