
import sys
sys.path.append('..')
import os
from mask_to_submission import masks_to_submission

ESSAI = 4

img_root = f'data/roadseg/test_res_{ESSAI}/'
fnames = [os.path.join(img_root, f) for f in os.listdir(img_root)]
masks_to_submission('essai1_submission.csv', *fnames)
