
import os
from PIL import Image
import numpy as np

SPLIT_LOGITS = "results/foursplit/logits"
SPLIT_LABEL = "results/foursplit/label"
SPLIT_OVERLAY = "results/foursplit/overlay"
PREFIXES = {
    SPLIT_LOGITS : "logit_",
    SPLIT_LABEL : "mask_",
    SPLIT_OVERLAY : "overlay_",
}

RECON_LOGITS = "results/foursplit/reconstructed_logits"
RECON_LABEL = "results/foursplit/reconstructed_label"
RECON_OVERLAY = "results/foursplit/reconstructed_overlay"

N_RECONSTRUCTED = 50
CROPS_BY_IMG = 4

AREAS = ((0,0,400,400),(0,208,400,608),(208,0,608,400),(208,208,608,608))

def save_L_nparr(save_dir, fname, arr):
    res_im = Image.fromarray(arr, 'L')
    res_im.save(os.path.join(save_dir, fname))

im_crops = [] 
for _ in range(N_RECONSTRUCTED):
    im_crops.append([])

curr_dir = SPLIT_LOGITS


# Create elementwise divisor matrix
divmat = np.array()
for area in AREAS:


# Maps a reconstructed image index to its 4 crops in im_crops
for fn in os.listdir(curr_dir):    
    if not "png" in fn: continue

    fpath = os.path.join(curr_dir, fn)
    crop_id = int(fn.replace(PREFIXES[curr_dir], '').replace(".png", ''))
    im = Image.open(fpath)
    # Puts crop i in image i // 4
    im_crops[(crop_id - 1) // CROPS_BY_IMG].append(np.array(im, dtype=np.float))

# Create a arr with 4 layers with 4 crops, and average the 4 layers
for im_idx in range(N_RECONSTRUCTED):
    assert len(im_crops[im_idx]) == 4
    layers = np.zeros((4, 608, 608))
    for i in range(4):
        x0,y0,x1,y1 = AREAS[i]
        print(i, layers[i,30,30])
        layers[i,x0:x1,y0:y1] = im_crops[im_idx][i]
        print(i, layers[i,30,30])
    result = layers.mean(axis=0)
    print("res", result[30,30])
    print(result.shape)
    save_L_nparr(RECON_LOGITS, str(im_idx) + ".png", result)

print(im_crops[0][0].shape)
