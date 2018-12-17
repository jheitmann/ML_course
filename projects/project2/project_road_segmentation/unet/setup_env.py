
import os
from shutil import copyfile

from common import GET_VERBOSE_PRINT, TRAIN_IMG_PATH, TRAIN_GT_PATH

"""
Methods to setup the required environement for the project
"""

ENV = {
    "data" : {
        "train" : {
            "image" : {},
            "label" : {},
            "aug" : {},
        },
        "test" : {
            "image" : {},
            "foursplit" : {},
        }
    },
    "results" : {
        "train" : {
            "label" : {},
            "logits" : {},
            "overlay" : {},
        },
        "test" : {
            "label" : {},
            "logits" : {},
            "overlay" : {},
        }
    }
}

def get_paths(envdic, acc_path=""):
    """
    Lists paths of envdic leaves folders
    Used recursively on ENV, lists all leaves folders paths in ENV.
    Args:
        envdic: mapping dirname -> [envdic|{}], envdic if dirname has subfolders, {} if leaf folder (see ENV ex.)
        acc_path: accumulator path of current envdic for os.path.join
    Returns:
        list of full paths from acc_path_0 to the leaves directory (ie only those without subfolders)
    """
    return [acc_path] if not envdic else sum(
        (get_paths(subdic, os.path.join(acc_path, dirname)) for dirname, subdic in envdic.items()), [])

def create_env(root_folder, *, verbose=False):
    """
    Creates ENV tree in root_folder. root_folder should be an EMPTY directory, or not exist at all.
    Args:
        root_folder: path to needed root folder of ENV tree, if inexistant, is created.
    Raises:
        EnvironementError: when an existing directory that should be created is found on disk
    """
    vprint = GET_VERBOSE_PRINT(verbose)
    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)
    for d in get_paths(ENV, acc_path=root_folder):
        if os.path.isdir(d):
            raise EnvironmentError(f"Found a pre-existing directory at {d}. Aborting.")
        vprint(f"create_env is making folder {d}")
        os.makedirs(d)

def check_env(root_folder, *, verbose=False):
    """
    Checks if the env at root_folder is complete
    Args:
        root_folder: The root folder of the required environement
        verbose: Prints optional debugging informations    
    Returns:
        The first missing folder path, or None in success case
    """
    vprint = GET_VERBOSE_PRINT(verbose)
    for d in get_paths(ENV, acc_path=root_folder):
        if not os.path.isdir(d):
            vprint(f"Could not find subdirectory {d}.")
            return d
        vprint(f"Found subdirectory {d}.")
    return None

def prepare_train(root_folder, *, verbose=False):
    """
    Prepares the structure necessary for running utrain.py from the original dataset
    Args:
        root_folder: The root folder of the required environement
        verbose: Prints optional debugging informations   
    Raises:
        AssertionError: when the training/ original dataset is not found in root_folder
    """
    
    vprint = GET_VERBOSE_PRINT(verbose)

    missing = check_env(root_folder)
    vprint(f"check_env on {root_folder} returned {missing}")
    if not missing:
        return


    assert os.path.isdir(os.path.join(root_folder, "training/")), f"The training/ dataset folder was not found in {root_folder}."
    create_env(root_folder, verbose=verbose)
    TRANSFER = copyfile # choose one between copyfile or os.rename for resp. copying or moving (moving will "destroy" original dataset folders)

    img_dir_path, gt_dir_path = (os.path.join(root_folder, "training", subf) for subf in ("images", "groundtruth"))
    vprint(f"Using images and groundtruth folders {img_dir_path}, {gt_dir_path}")

    for fimg in os.listdir(img_dir_path):        
        fpath = os.path.join(img_dir_path, fimg)
        new_path = os.path.join(TRAIN_IMG_PATH, fimg)
        vprint(f"Moving file {fpath} to {new_path}")
        TRANSFER(fpath, new_path)

    for fgt in os.listdir(gt_dir_path):        
        fpath = os.path.join(gt_dir_path, fgt)
        new_path = os.path.join(TRAIN_GT_PATH, fgt)
        vprint(f"Moving file {fpath} to {new_path}")
        TRANSFER(fpath, new_path)

def prepare_test(root_folder, *, verbose=False):
    """
    Prepares the structure necessary for running utrain.py from the original dataset
    Args:
        root_folder: The root folder of the required environement
        verbose: Prints optional debugging informations   
    Raises:
        AssertionError: when the training/ original dataset is not found in root_folder
    """

    vprint = GET_VERBOSE_PRINT(verbose)

    missing = check_env(root_folder)
    vprint(f"check_env on {root_folder} returned {missing}")
    if not missing:
        return

    assert os.path.isdir(os.path.join(root_folder, "test_set_images/")), f"The test_set_images/ dataset folder was not found in {root_folder}."
    create_env(root_folder, verbose=verbose)

    
