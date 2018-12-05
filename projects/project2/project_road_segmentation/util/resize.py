
import sys
import os
from PIL import Image

def resize(input_folder, output_folder='output_resize/', new_size=(256, 256)):
    os.mkdir(output_folder)
    for f in os.listdir(input_folder):
        im = Image.open(os.path.join(input_folder, f))
        im2 = im.resize((int(new_size[0]), int(new_size[1])), Image.ANTIALIAS)
        nfname = f.split('.')[0] + '_resized.' + f.split('.')[1]
        print('resized', nfname)
        im2.save(os.path.join(output_folder, nfname))

if __name__=="__main__":
    """
    Script console usage example :
    python resize.py old_images 512 512 new_images
    """
    _, input_folder, w, h, output_folder = sys.argv
    resize(input_folder, output_folder, (w, h))
