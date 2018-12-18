import cv2
import numpy as np
import os
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from common import PIXEL_DEPTH, RESULTS_PATH


def extract_data(image_path, num_images, img_height, as_rgb, verbose=True):
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
    img_filenames = [os.path.join(image_path, fn) for fn in os.listdir(image_path)]
    img_filenames.sort()

    for filename in img_filenames[:num_images]: 
        if verbose:
            print(f"Loading {filename}")
        img = cv2.imread(filename, as_rgb)
        img = cv2.resize(img, dsize=(img_height, img_height), interpolation=cv2.INTER_AREA)
        if not as_rgb:
            img = img[..., np.newaxis]
        img = img.astype('float32')
        img /= PIXEL_DEPTH
        imgs.append(img)            

    return np.array(imgs)

def extract_labels(label_path, num_images, img_height, verbose=True):
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
    gt_filenames = [os.path.join(label_path, fn) for fn in os.listdir(label_path)]
    gt_filenames.sort()

    for filename in gt_filenames[:num_images]:
        if verbose:
            print (f"Loading {filename}")
        labels = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        labels = cv2.resize(labels, (img_height, img_height))
        labels = labels[..., np.newaxis]
        labels = labels.astype('float32')
        labels /= PIXEL_DEPTH
        labels[labels >= 0.5] = 1
        labels[labels < 0.5] = 0
        gt_imgs.append(labels)

    return np.array(gt_imgs)

def get_checkpoint(img_height, rgb, monitor):
    hdf5_name = "unet_{}_{}_{}.hdf5".format("rgb" if rgb else "bw", img_height, str(datetime.now()).replace(':', '_').replace(' ', '_'))
    print("hdf5 name:", hdf5_name)
    
    ckpt_file = os.path.join(RESULTS_PATH, hdf5_name)
    return ModelCheckpoint(ckpt_file, monitor=monitor, verbose=1, save_best_only=True)

def split_data(x, y, ratio, seed=1):
    """
    Splits the dataset based on the split ratio, uses seed for the random selection of indices
    Args:
        x: data x
        y: labels y    
        ratio: train set will be 100*ratio % of the original, test 100*(1-ratio) %
        seed: rng seed
    Returns:
        4-tuple (x_train, x_test, y_train, y_test)
    """    
    # set seed
    np.random.seed(seed)
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
    mask_save_prefix="mask", save_to_dir=None, shuffle=False, seed=1):
    """
    Args:
    batch_size
        train_path: path to directory containing subdirectories of images and masks
        image_folder: name of subdirectory in train_path containing images
        mask_folder: name of subdirectory in train_path containing masks
        data_gen_args: args dict fed to the ImageDataGenerator objects
        target_size: resizing size for both images and labels
        color_mode: [grayscale|rbg|rgba] the generator will load resp. 1, 3 or 4 channels
        interpolation: [nearest|bilinear|bicubic|lanczos|box|hamming] method for resampling to target_size
        image_save_prefix: save_prefix of flow_from_directory for images
        mask_save_prefix: save_prefix of flow_from_directory for images
        save_to_dir: [None|str] path of directory in which will be saved the generated pictures. None disables saving
        shuffle: bool set to True to shuffle the flow from the the folders
        seed: rng seed used for shuffling and random transformations
    Raises:
        AssertionError: when any subfolder name ends with a separator char (not supported as classes)
    Returns:
        A generator function generating a formated tuple (image, label) of np.array
    """
    
    image_datagen, mask_datagen = ImageDataGenerator(**data_gen_args), ImageDataGenerator(**data_gen_args)

    # Makes flows
    assert not image_folder.endswith(os.path.sep) and not image_folder.endswith('/'),\
        f"The image path {image_folder} must NOT end with separator for some reason (ex: image/ -> image)"
    assert not mask_folder.endswith(os.path.sep) and not mask_folder.endswith('/'),\
        f"The label path {mask_folder} must NOT end with separator for some reason (ex: label/ -> label)"

    # If save_to_dir is provided, will pass save_to_dir+subf to generators, otherwise doesn't pass this param.
    param_save_to = lambda subf: dict(save_to_dir=os.path.join(save_to_dir, subf)) if save_to_dir else {}

    train_image_generator = image_datagen.flow_from_directory(
        train_path,
        batch_size=batch_size,
        classes=[image_folder],
        class_mode=None,
        target_size=target_size,
        color_mode=color_mode,
        interpolation=interpolation,
        **param_save_to("train"),
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
        **param_save_to("train"),
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
        **param_save_to("val"),
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
        **param_save_to("val"),
        save_prefix="val_"+mask_save_prefix,
        shuffle=shuffle,
        seed=seed,
        subset="validation")

    # Makes the generator function of tuples using the two flows
    def generator(images, labels):
        for (image, label) in zip(images, labels):
            yield convert_01(image, label)

    return generator(train_image_generator, train_mask_generator), generator(validation_image_generator, validation_mask_generator)

"""
def listdirpaths(dirpath):
    \"""
    Args:
        dirpath: path to directory containing files
    Returns:
        a list of strings "{dirpath}/{filename}" for each file in dirpath/
    \"""
    return [os.path.join(f) for f in os.listdir(dirpath)]

def get_train_generator(batch_size,train_path,image_folder,label_folder,aug_dict,
    image_color_mode="rgb",label_color_mode="grayscale",
    image_save_prefix="image",mask_save_prefix="mask",
    save_to_dir=None,target_size=(400,400),seed=1):
    \"""
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
    \"""
    
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
"""