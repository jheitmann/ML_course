
import os

#utrain
SPLIT_TRAIN_PATH = "data/split/training"
SPLIT_VAL_PATH = "data/split/validation"
TRAIN_PATH = "data/train"
TEST_PATH = "data/test"
IMG_SUBFOLDER = "image"
GT_SUBFOLDER = "label"

N_SPLIT_TRAIN = 80
N_SPLIT_VAL = 20
N_TRAIN_IMAGES = 100
AUG_SAVE_PATH = "data/train/aug/"

TRAIN_IMG_PATH = os.path.join(TRAIN_PATH, IMG_SUBFOLDER)
TRAIN_GT_PATH = os.path.join(TRAIN_PATH, GT_SUBFOLDER)

SPLIT_TRAIN_IMG_PATH = os.path.join(SPLIT_TRAIN_PATH, IMG_SUBFOLDER)
SPLIT_TRAIN_GT_PATH = os.path.join(SPLIT_TRAIN_PATH, GT_SUBFOLDER)
SPLIT_VAL_IMG_PATH = os.path.join(SPLIT_VAL_PATH, IMG_SUBFOLDER)
SPLIT_VAL_GT_PATH = os.path.join(SPLIT_VAL_PATH, GT_SUBFOLDER)

#utest
TEST_IMG_PATH = os.path.join(TEST_PATH, IMG_SUBFOLDER)

TESTING_PATH_FOURSPLIT = "data/test/foursplit/"
RESULTS_PATH = "results/"
SUBM_PATH = "results/output.csv"
N_TEST_IMAGES = 50
TEST_IMG_HEIGHT = 608
TRAIN_IMG_HEIGHT = 400

#pre/postprocessing
PIXEL_DEPTH = 255
IMG_PATCH_SIZE = 16
PIXEL_THRESHOLD = 127
PREDS_PER_IMAGE = 4
AREAS = ((0,0,400,400),(208,0,608,400),(0,208,400,608),(208,208,608,608))

# returns print if verbose==True, otherwise an invisible function w. same signature
GET_VERBOSE_PRINT = lambda verbose: (lambda *a, **kwa: print(*a, **kwa) if verbose else None)
