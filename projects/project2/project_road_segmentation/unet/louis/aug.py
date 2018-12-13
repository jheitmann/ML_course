 
import sys
sys.path.append('..')
import os
from keras.preprocessing.image import ImageDataGenerator

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

def main():
    pass

if __name__=="__main__":
    main()
