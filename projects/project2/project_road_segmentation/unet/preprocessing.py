import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

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

def convert_01(image, label):
    """
    Converts an img and mask from values in [0;255] to [0;1]
    Args:
        image: The image numpy array
        label: The mask numpy array
    Returns:
        Converted image and label
    """
    image /= 255.
    label /= 255.
    label[label <= .5], label[label > .5] = 0, 1
    return image, label

def listdirpaths(dirpath):
    """
    Args:
        dirpath: path to directory containing files
    Returns:
        a list of strings "{dirpath}/{filename}" for each file in dirpath/
    """
    return [os.path.join(f) for f in os.listdir(dirpath)]

def getTrainGenerator(batch_size,train_path,image_folder,label_folder,aug_dict,
    image_color_mode="rgb",label_color_mode="grayscale",
    image_save_prefix="image",mask_save_prefix="mask",
    save_to_dir=None,target_size=(400,400),seed=1):
    """
    Args:
        train_path: path to directory containing subdirectories of images and labels
        image_folder: name of subdirectory in train_path containing images
        label_folder: name of subdirectory in train_path containing labels
        *_color_mode: color mode of *
        save_to_dir: path of directory in which will be saved the generated pictures
        target_size: resizing size for both images and labels
        seed: rng seed
    Returns:
        A generator function generating a formated tuple (image, label) of np.array
    """
    
    # Makes ImageDataGenerators according to aug_dict
    image_datagen, label_datagen = ImageDataGenerator(**aug_dict), ImageDataGenerator(**aug_dict)

    # Makes flows
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    
    # Makes the generator function of tuples using the two flows
    def generator():
        for (image, label) in zip(image_generator, label_generator):
            yield convert_01(image, label)

    return generator