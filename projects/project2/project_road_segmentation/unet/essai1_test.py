
import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from model import unet, ModelCheckpoint
from data import testGeneratorAlt, saveResult, saveResultAlt, labelVisualize

CKPT_FILENAME = "essai1.hdf5"
TEST_FOLDER = "data/roadseg/testLS"
N_TESTS = 50

model = unet()
model_checkpoint = ModelCheckpoint(filepath=CKPT_FILENAME, verbose=1, save_best_only=True)
model.load_weights(CKPT_FILENAME)
#testGene = testGeneratorAlt(TEST_FOLDER)

TARGET_SIZE = (256,256)
FLAG_MULTI_CLASS = False

for i, f in enumerate(os.listdir(TEST_FOLDER)[:N_TESTS]):
    print('prediction #', i, f)
    img = io.imread(os.path.join(TEST_FOLDER, f))
    img = img / 255
    img = trans.resize(img, TARGET_SIZE)
    img = np.reshape(img, img.shape + (1,)) if (not FLAG_MULTI_CLASS) else img
    img = np.reshape(img, (1,) + img.shape)
    print((1,) + img.shape)

    result = model.predict(img, verbose=1)
    print(result.shape)
    resname = 'pred_'+f
    print('saving', resname)
    saveResultAlt(TEST_FOLDER, result, [resname])
