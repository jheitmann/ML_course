
from model import unet, ModelCheckpoint
from data import trainGenerator, testGeneratorAlt, saveResult

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

BATCH_SIZE = 2
DS_PATH = "data/roadseg/train"
IMG_SUBFOLDER = "image"
MSK_SUBFOLDER = "label"
AUG_SUBFOLDER = f"{DS_PATH}/aug/"
CKPT_FILENAME = "essai1.hdf5"
N_STEPS_PER_EPOCH = 10
N_EPOCH = 1
myGene = trainGenerator(BATCH_SIZE, DS_PATH, IMG_SUBFOLDER, MSK_SUBFOLDER, data_gen_args, image_color_mode="rgb", target_size=(400, 400), save_to_dir=AUG_SUBFOLDER)

model = unet(input_size=(400,400,3))
model_checkpoint = ModelCheckpoint(CKPT_FILENAME, monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=N_STEPS_PER_EPOCH, epochs=N_EPOCH, callbacks=[model_checkpoint])
