
import os
import random

IMAGES_DIR = "datasets/project_dataset/training/images/"

# np out of n images will be used for training and n(1-p) for testing
p = .8

train_imgset, test_imgset = [], []
images = os.listdir(IMAGES_DIR)
random.shuffle(images)
train_n = int(len(images) * p)
print(images)