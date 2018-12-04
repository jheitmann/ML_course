
import sys
import os
from PIL import Image

os.mkdir(sys.argv[2])
for f in os.listdir(sys.argv[1]):
    print(f)
    im = Image.open(sys.argv[1] + '/' + f)
    iml = im.convert('L')
    iml.save(sys.argv[2] + '/' + f)
