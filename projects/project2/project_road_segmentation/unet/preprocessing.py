import cv2
import numpy as np
import os


PIXEL_DEPTH = 255
IMG_PATCH_SIZE = 16

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images, img_height, as_rgb):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0, 1].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = cv2.imread(image_filename, as_rgb)
            img = cv2.resize(img, dsize=(img_height, img_height), interpolation=cv2.INTER_AREA)
            if not as_rgb:
                img = img[..., np.newaxis]
            img = img.astype('float32')
            img /= PIXEL_DEPTH
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    img_width = imgs[0].shape[0]
    img_height = imgs[0].shape[1]

    return np.array(imgs)
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images, img_height):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            labels = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            labels = cv2.resize(labels, (img_height, img_height))
            labels = labels[..., np.newaxis]
            labels = labels.astype('float32')
            labels /= PIXEL_DEPTH
            labels[labels >= 0.5] = 1
            labels[labels < 0.5] = 0
            gt_imgs.append(labels)
        else:
            print ('File ' + image_filename + ' does not exist')

    return np.array(gt_imgs)

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels
