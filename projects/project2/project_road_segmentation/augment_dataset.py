from PIL import Image, ImageOps
import os

PATHS = ["datasets/roadseg/images", "datasets/roadseg/groundtruth"]
IMGEXT = ".png"
DEBUG = True

debug = lambda *a, **k: print(*a, **k) if DEBUG else None

def enum_files_in(paths):
    for p in paths:
        for f in os.listdir(p):
            yield os.path.join(p, f)

def rotate_all(dirpaths):
    for fpath in enum_files_in(dirpaths):
        im = Image.open(fpath)
        for a in [90, 180, 270]:
            ri = im.rotate(a, expand=1)
            debug(f'Saving rotated {a} deg image', fpath.replace(IMGEXT, f"rotated{a}{IMGEXT}"))
            ri.save(fpath.replace(IMGEXT, f"rotated{a}{IMGEXT}"))

def mirror_all(dirpaths):
    for fpath in enum_files_in(dirpaths):
        im = Image.open(fpath)
        mi = ImageOps.mirror(im)
        debug('Saving mirrored image', fpath.replace(IMGEXT, f"mirrored{IMGEXT}"))
        mi.save(fpath.replace(IMGEXT, f"mirrored{IMGEXT}"))

def augment(dirpaths):
    mirror_all(dirpaths)
    rotate_all(dirpaths)

def rm_augmented(dirpaths):
    for fpath in enum_files_in(dirpaths):
        if "rotated" in fpath or "mirrored" in fpath:
            debug('Removing old augmented image', fpath)
            os.remove(fpath)

rm_augmented(PATHS)
augment(PATHS)
