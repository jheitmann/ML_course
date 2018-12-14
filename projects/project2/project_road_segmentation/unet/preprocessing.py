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

def extract_data(image_path, image_prefix, num_images, img_height, as_rgb, *, verbose=True):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0, 1].
    Args:
        image_path: image folder path
        image_prefix: image name prefix, typically satImage_ or test_
        num_images: size 0 of returned tensor, ammount of extracted images
        img_height: resized image target width/height
        as_rgb: flag set True when images need to be loaded as rgb
        verbose: named flag set False to disable information printed by this method
    Raises:
        FileNotFoundError: if no image {filename}{id}.png is found for id in [1; num_images] 
    Returns:
        4D tensor [image index, y, x, channels]
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = image_prefix + ("%.3d" % i)
        image_filename =  os.path.join(image_path, f"{imageid}.png")
        if not os.path.isfile(image_filename):
            raise FileNotFoundError(f"File {image_filename} does not exist.") 
        if verbose:
            print(f"Loading {image_filename}")
        img = cv2.imread(image_filename, as_rgb)
        img = cv2.resize(img, dsize=(img_height, img_height), interpolation=cv2.INTER_AREA)
        if not as_rgb:
            img = img[..., np.newaxis]
        img = img.astype('float32')
        img /= PIXEL_DEPTH
        imgs.append(img)            

    # Are those lines useful?
    num_images = len(imgs)
    img_width = imgs[0].shape[0]
    img_height = imgs[0].shape[1]

    return np.array(imgs)

def value_to_class(v, foreground_threshold=0.25):
    """
    Assign a label to a patch v
    Args:
        v: patch np.array
        foreground_threshold: percentage of pixels > 1 required to assign a foreground label to a patch
    Returns:
        one-hot vector of corresponding class
    """
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

def extract_labels(label_path, num_images, img_height, *, verbose=True):
    """
    Extract the labels into a 1-hot matrix [image index, label index].
    Args:
        label_path: path to label folder
        num_images: size 0 of returned tensor, ammount of extracted images
        img_height: resized image target width/height
        verbose: named flag set False to disable information printed by this method
    Raises:
        FileNotFoundError: if no image {filename}{id}.png is found for id in [1; num_images] 
    Returns:
        1-hot matrix [image index, label index]
    """
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = os.path.join(label_path, f"{imageid}.png")
        if not os.path.isfile(image_filename):
            raise FileNotFoundError(f"File {image_filename} does not exist.")
        if verbose:
            print (f"Loading {image_filename}")
        labels = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
        labels = cv2.resize(labels, (img_height, img_height))
        labels = labels[..., np.newaxis]
        labels = labels.astype('float32')
        labels /= PIXEL_DEPTH
        labels[labels >= 0.5] = 1
        labels[labels < 0.5] = 0
        gt_imgs.append(labels)

    return np.array(gt_imgs)

def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Convert array of labels to an image
    Args:
        imgwidth: width of returned image
        imgheight: height of returned image
        w: horizontal step
        h: vertical step
        labels: logit np.array
    Returns:

    """
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

def split_data(x, y, ratio, myseed=1):
    """ Splits the dataset based on the split ratio, uses myseed for the random selection of indices """    
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

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

def get_generators(batch_size, train_path, image_folder, mask_folder, data_gen_args, 
    target_size=(400,400), color_mode="rgb", interpolation="lanczos", image_save_prefix="image", 
    mask_save_prefix="mask", save_to_dir=None, shuffle=True, seed=1):
    
    image_datagen, mask_datagen = ImageDataGenerator(**data_gen_args), ImageDataGenerator(**data_gen_args)

    # Makes flows
    assert not image_folder.endswith(os.path.sep) and not image_folder.endswith('/'),\
        f"The image path {image_folder} must NOT end with separator for some reason (ex: image/ -> image)"
    assert not mask_folder.endswith(os.path.sep) and not mask_folder.endswith('/'),\
        f"The label path {mask_folder} must NOT end with separator for some reason (ex: label/ -> label)"

    train_image_generator = image_datagen.flow_from_directory(
        train_path,
        batch_size=batch_size,
        classes=[image_folder],
        class_mode=None,
        target_size=target_size,
        color_mode=color_mode,
        interpolation=interpolation,
        save_to_dir=save_to_dir+"train",
        save_prefix=image_save_prefix,
        shuffle=shuffle,
        seed=seed,
        subset="training")
    train_mask_generator = mask_datagen.flow_from_directory(
        train_path,
        batch_size=batch_size,
        classes=[mask_folder],
        class_mode=None,
        target_size=target_size,
        color_mode="grayscale",
        interpolation=interpolation,
        save_to_dir=save_to_dir+"train",
        save_prefix=mask_save_prefix,
        shuffle=shuffle,
        seed=seed,
        subset="training")
    validation_image_generator = image_datagen.flow_from_directory(
        train_path,
        batch_size=batch_size,
        classes=[image_folder],
        class_mode=None,
        target_size=target_size,
        color_mode=color_mode,
        interpolation=interpolation,
        save_to_dir=save_to_dir+"val",
        save_prefix="val_"+image_save_prefix,
        shuffle=shuffle,
        seed=seed,
        subset="validation")
    validation_mask_generator = mask_datagen.flow_from_directory(
        train_path,
        batch_size=batch_size,
        classes=[mask_folder],
        class_mode=None,
        target_size=target_size,
        color_mode="grayscale",
        interpolation=interpolation,
        save_to_dir=save_to_dir+"val",
        save_prefix="val_"+mask_save_prefix,
        shuffle=shuffle,
        seed=seed,
        subset="validation")

    # Makes the generator function of tuples using the two flows
    def generator(images, labels):
        for (image, label) in zip(images, labels):
            yield convert_01(image, label)

    return generator(train_image_generator, train_mask_generator), generator(validation_image_generator, validation_mask_generator)

def listdirpaths(dirpath):
    """
    Args:
        dirpath: path to directory containing files
    Returns:
        a list of strings "{dirpath}/{filename}" for each file in dirpath/
    """
    return [os.path.join(f) for f in os.listdir(dirpath)]

def get_train_generator(batch_size,train_path,image_folder,label_folder,aug_dict,
    image_color_mode="rgb",label_color_mode="grayscale",
    image_save_prefix="image",mask_save_prefix="mask",
    save_to_dir=None,target_size=(400,400),seed=1):
    """
    Args:
        train_path: path to directory containing subdirectories of images and labels
        image_folder: name of subdirectory in train_path containing images
        label_folder: name of subdirectory in train_path containing labels
        *_color_mode: color mode of *
        *_save_prefix: save_prefix of flow_from_directory of * 
        save_to_dir: path of directory in which will be saved the generated pictures
        target_size: resizing size for both images and labels
        seed: rng seed
    Returns:
        A generator function generating a formated tuple (image, label) of np.array
    """
    
    # Makes ImageDataGenerators according to aug_dict
    image_datagen, label_datagen = ImageDataGenerator(**aug_dict), ImageDataGenerator(**aug_dict)

    # Makes flows
    assert not image_folder.endswith(os.path.sep) and not image_folder.endswith('/'),\
        f"The image path {image_folder} must NOT end with separator for some reason (ex: image/ -> image)"
    assert not label_folder.endswith(os.path.sep) and not label_folder.endswith('/'),\
        f"The label path {label_folder} must NOT end with separator for some reason (ex: label/ -> label)"
    subset_train, subset_val = (dict(subset="training"), dict(subset="validation")) if "validation_split" in aug_dict else ({}, {})
    train_image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        **subset_train)
    train_label_generator = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        **subset_train)
    validation_image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        **subset_val)
    validation_label_generator = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        **subset_val)
    # Makes the generator function of tuples using the two flows
    def generator(images, labels):
        for (image, label) in zip(images, labels):
            yield convert_01(image, label)

    return generator(train_image_generator, train_label_generator), generator(validation_image_generator, validation_label_generator)
