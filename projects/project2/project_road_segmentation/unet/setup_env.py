
import os

"""
Methods to setup the required environement for the project
"""

ENV = {
    "data" : {
        "train" : {
            "image" : {},
            "label" : {},
        },
        "test" : {
            "image" : {},
            "foursplit" : {},
        },
    },
    "results" : {

    },
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

def create_env(root_folder):
    """
    Creates ENV tree in root_folder. root_folder should be an EMPTY directory, or not exist at all.
    Args:
        root_folder: path to needed root folder of ENV tree, if inexistant, is created.
    Raises:
        EnvironementError: when an existing directory that should be created is found on disk
    """
    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)
    for d in get_paths(ENV, acc_path=root_folder):
        if os.path.isdir(d):
            raise EnvironmentError(f"Found a pre-existing directory at {d}. Aborting.")
        os.makedirs(d)

def check_env(root_folder, *, verbose=False):
    """ Checks if the env at root_folder is complete """
    vprint = lambda *a, **kwa: print(*a, **kwa) if verbose else None
    for d in get_paths(ENV, acc_path="."):
        if not os.path.isdir(d):
            vprint(f"Could not find subdirectory {d}.")
            return False        
        vprint(f"Found subdirectory {d}.")
    return True
