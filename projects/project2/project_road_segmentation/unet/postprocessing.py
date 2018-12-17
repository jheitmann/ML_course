import cv2
import matplotlib.image as mpimg
import numpy as np
import os

from PIL import Image
from common import PIXEL_DEPTH, IMG_PATCH_SIZE, PIXEL_THRESHOLD, PREDS_PER_IMAGE, AREAS

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])

def write_predictions_to_file(predictions, labels, filename):
    """
    Write predictions from neural network to a file
    Args:
        predictions: np.array with predictions
        labels: labels
        filename: name of file to be written into
    """
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

def print_predictions(predictions, labels):
    """
    Print predictions and labels
    """
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

def img_float_to_uint8(img):
    """
    Converts the img np.array to corresponding uint8 np.array
    Args:
        img: float np.array representing the img
    Returns:
        resulting np.array
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    """
    Overlays img and predicted_img to have a superposed visualisation
    Args:
        img: image
        predicted_img: label/groundtruth
    Returns:
        overlayed image
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img # *PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def get_prediction_with_groundtruth(file_name, img_prediction):
    """
    Get a concatenation of the prediction and groundtruth for given input file
    """
    img = mpimg.imread(file_name)
    cimg = concatenate_images(img, img_prediction)

    return cimg

def get_prediction_with_overlay(img_filename, img_prediction):
    """
    Get prediction overlaid on the original image for given input file
    """
    img = mpimg.imread(img_filename)
    oimg = make_img_overlay(img, img_prediction)

    return oimg

def four_split_mean(masks, output_height):
    """
    Transforms the 4*N split predictions to N aggregated predictions, averaging four splits into one composite
    Args:
        masks: np.array of shape [4*N, SPLIT_PRED_SIZE, SPLIT_PRED_SIZE]
        output_height: size of output composite prediction masks
    Returns:
        np.array of shape [N, OUTPUT_HEIGHT, OUTPUT_HEIGHT] 
    """
    num_preds = masks.shape[0]
    n_imgs = int(num_preds / PREDS_PER_IMAGE)
    preds_height = masks.shape[1]
    print(f"Input shape: {masks.shape}")
    grouped_preds = masks.reshape((n_imgs, PREDS_PER_IMAGE, preds_height, preds_height))
    print(f"Grouped shape: {grouped_preds.shape}")
    
    averaged_preds = []
    onemat = np.ones((preds_height,preds_height), dtype=np.uint8)
    divmat = np.zeros((output_height,output_height), dtype=np.uint8)
    for area in AREAS:
        x0,y0,x1,y1 = area
        divmat[x0:x1,y0:y1] += onemat

    for i in range(n_imgs):
        four_preds = grouped_preds[i]
        output = np.zeros((output_height,output_height))
        
        for partial_pred_idx in range(PREDS_PER_IMAGE):
            partial_pred = four_preds[partial_pred_idx]
            x0,y0,x1,y1 = AREAS[partial_pred_idx]
            output[x0:x1,y0:y1] += partial_pred

        output = (output / divmat).astype("uint8")

        averaged_preds.append(output)

    return np.array(averaged_preds)

def convert_predictions(logits_masks, output_height, four_split, averaged_preds_size,
    mask_path, logits_path, overlay_path, test_name, save_logits, save_overlay):
    """
    Converts preds into image masks and serializes them (and optionaly also logits & overlay)
    Args:
        logits_masks: np.array of logit masks
        output_height: size for resizing logits into (if not using four_split)
        four_split: bool set True to use four_split method instead of resizing (more details in report)
        averaged_preds_size: size of outputs fed into four_split algorithm        
        mask_path: path the the folder containing resulting masks
        logits_path: path the the folder containing resulting logits (if using save_logits=True)
        test_name: a partial path pointing to image before "_number.png", like "data/test/test"
        save_logits: bool set True to save the logits images
        save_overlay: bool set True to save the overlays of the masks in red transparency over the imgs
    Returns:
        list of paths to the predicted masks files
    """

    num_preds = logits_masks.shape[0]
    logits_masks_scaled = np.zeros((num_preds,output_height,output_height))
    for i in range(logits_masks.shape[0]):
        logits_masks_scaled[i] = cv2.resize(logits_masks[i], dsize=(output_height,output_height), 
                                                interpolation=cv2.INTER_CUBIC)

    if four_split:
        logits_masks_scaled = four_split_mean(logits_masks_scaled, averaged_preds_size)

    predicted_mask_files = []
    predicted_masks_scaled = np.zeros(logits_masks_scaled.shape)
    predicted_masks_scaled[logits_masks_scaled > PIXEL_THRESHOLD] = 255
    
    for i in range(1, logits_masks_scaled.shape[0]+1):
        filename = "_%.3d" % i

        logits_mask_scaled = logits_masks_scaled[i-1]
        predicted_mask_scaled = predicted_masks_scaled[i-1]

        if save_logits:
            logits_relative_path = logits_path + "logit" + filename + ".png"
            cv2.imwrite(logits_relative_path, logits_mask_scaled)   

        mask_relative_path = mask_path + "mask" + filename + ".png"
        print ('Predicting ' + mask_relative_path)
        cv2.imwrite(mask_relative_path, predicted_mask_scaled)
        predicted_mask_files.append(mask_relative_path)
        
        if save_overlay:
            overlay_relative_path = overlay_path + "overlay" + filename + ".png"
            test_relative_path = test_name + filename + ".png"
            oimg = get_prediction_with_overlay(test_relative_path, predicted_mask_scaled)
            oimg.save(overlay_relative_path)
    
    return predicted_mask_files
    

def predictions_to_masks(result_path, test_name, preds, output_height, four_split, 
    averaged_preds_size, mask_folder="label/", logits_folder='logits/', 
    overlay_folder='overlay/', save_logits=True, save_overlay=True):
    """
    Converts predictions to logits and call convert_logits on logits to generate masks images,
    logits images, overlays images, and returns the ouput of convert_logits.
    Args:
        result_path: path to root of masks, logits, overlays subfolders
        test_name: a partial path pointing to image before "_number.png", like "data/test/test"
        preds: np.array of predictions sized [N_PREDICTIONS, PRED_SIZE, PRED_SIZE] containing logits in [0;1]
        output_height: size for resizing logits into (if not using four_split)
        four_split: bool set True to use four_split method instead of resizing (more details in report)
        averaged_preds_size: size of outputs fed into four_split algorithm        
        mask_folder: name of subfolder containing resulting masks
        logits_folder: name of subfolder containing resulting logits (if using save_logits=True)
        overlay_folder: name of subfolder containing resulting overlays (if using save_overlay=True) 
        save_logits: bool set True to save the logits (non-binary pixel intensities) to logits_folder
        save_overlay: bool set True to save the overlays (mask in red transparency over the img) to overlay_folder
    Returns:
        list of paths to the predicted masks files
    """
    mask_path = os.path.join(result_path, mask_folder)
    logits_path = os.path.join(result_path, logits_folder)
    overlay_path = os.path.join(result_path, overlay_folder)

    #predicted_masks = np.zeros(preds.shape)
    #predicted_masks[preds >= P_THRESHOLD] = 1.0 

    logits_masks = preds * PIXEL_DEPTH
    #predicted_masks = predicted_masks * PIXEL_DEPTH

    logits_masks = np.round(logits_masks).astype('uint8')
    #predicted_masks = predicted_masks.astype('uint8')

    logits_masks = np.squeeze(logits_masks)
    #predicted_masks = np.squeeze(predicted_masks)
 
    return convert_predictions(logits_masks, output_height, four_split, averaged_preds_size, mask_path,
                                    logits_path, overlay_path, test_name, save_logits, save_overlay)


def patch_to_label(patch, foreground_threshold=0.25):
    """
    Assign a label to a patch
    """
    df = np.mean(patch)
    return int(df > foreground_threshold)

def mask_to_submission_strings(image_filename, foreground_threshold=0.25):
    """
    Reads a single image and outputs the strings that should go into the submission file
    Args:
        image_filename: filename of mask image 
        path_size: patch size (w, h). Always 16
    Returns:
        yield all strings that should be serialized into the csv file corresp. to this image
    """
    # Get image number. Works on any image_filename like */*_1.* or */*_001.* for example.
    img_name = image_filename.split('/')[-1]
    img_number = int(img_name.split('_')[1].split('.')[0])
    
    # Read mask into np.array
    im = mpimg.imread(image_filename)
    for j in range(0, im.shape[1], IMG_PATCH_SIZE):
        for i in range(0, im.shape[0], IMG_PATCH_SIZE):
            # Get patch np.array from image np.array
            patch = im[i:i + IMG_PATCH_SIZE, j:j + IMG_PATCH_SIZE]
            # Convert to corresp. label
            label = patch_to_label(patch, foreground_threshold)
            # Yield resulting string
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

def masks_to_submission(submission_filename, image_filenames, foreground_threshold=0.25):
    """
    Converts images into a submission file.
    image_filenames must contain strs *_NUM.*, with NUM a str convertible to int
    ex: data/test/test_1.PNG or my/img/path/xyz_000000.jpeg are VALID filenames, but test2 is INVALID (no underscore, no .ext)
    Args:
        submission_filename: filename (path) of csv file created by this function
        image_filenames: iterator of masks filenames (paths) to convert to csv
    """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, foreground_threshold))

def compute_trainset_f1(test_csv, train_masks_dir="data/train/label", verbose=False):
    """
    Computes the f1 score of a result.csv resulting from a test on training data based on known train masks
    Args:
        test_csv: path to result csv of a test.py run on the TRAINING dataset
        train_masks_dir: path to masks images of training dataset
        verbose: set True for debugging information prints
    Returns:
        f1 score
    """
    vprint = lambda *a, **kwa: print(*a, **kwa) if verbose else None
    train_masks_filenames = [os.path.join(train_masks_dir, fn) for fn in os.listdir(train_masks_dir)]
    train_masks_filenames.sort()
    TRAIN_MASKS_CSV = "results/trainset_masks.csv"
    # Convert training masks to csv submission file at TRAIN_MASKS_CSV
    masks_to_submission(TRAIN_MASKS_CSV, train_masks_filenames)
    vprint(f"Saved training masks csv at {TRAIN_MASKS_CSV}")
    # Load csv lines
    with open(test_csv, "r") as f_test:
        with open(TRAIN_MASKS_CSV) as f_train:
            test_lines, train_lines = f_test.readlines()[1:], f_train.readlines()[1:]
    # Remove \n and space char from lines
    test_lines, train_lines = (
        [l.replace('\n', '').replace(' ', '') for l in lines]
        for lines in (test_lines, train_lines)
    )
    # Insure the csvs have one line for each patch in 100 images sized 400px*400px
    assert len(test_lines) == len(train_lines) == 100 * 25 * 25,\
        f"{len(test_lines)}, {len(train_lines)}, {100 * 25 * 25} not all equal."
    vprint(f"Sizes of test csv and train csv are both correctly equal to {100 * 25 * 25}")
    # Record true and false positives and negatives
    accs = ([], [], [], [])
    (tp, fp, tn, fn) = accs
    for train_l, test_l in zip(train_lines, test_lines):
        # Get prediction of train, test patch
        (train_coord, train_pred), (test_coord, test_pred) = (l.split(',') for l in (train_l, test_l))
        # Classify patch
        (
            tp if train_pred == test_pred == '1' else
            tn if train_pred == test_pred == '0' else
            fp if train_pred == '0' else
            fn
        ).append(test_coord)
    ntp, nfp, ntn, nfn = (len(acc) for acc in accs)
    vprint(f"n true pos={ntp}, n false pos={nfp}, n true neg={ntn}, n false neg={nfn}")
    assert sum(len(a) for a in accs) == 100 * 25 * 25,\
        f"Abnormal sum of true/false pos/neg ({sum(len(a) for a in accs)})"

    precision = ntp / (ntp + nfp)
    vprint("precision", precision)
    recall = ntp / (ntp + nfn)
    vprint("recall", recall)
    f1_score = 2/(1/precision + 1/recall)
    return f1_score

def gen_four_split(original_images_dir, foursplit_dir):
    """
    Draft implementation of four_split method
    """
    fnames = os.listdir(original_images_dir)
    fnames.sort()
    for fn in fnames:
        if not "png" in fn: continue
        original_index = int(fn.replace("test_", '').replace(".png", ''))
        fpath = os.path.join(original_images_dir, fn)
        print(fpath)
        oim = Image.open(fpath)
        oim_name = os.path.basename(fpath)
        crops = [oim.crop(area) for area in ((0,0,400,400),(0,208,400,608),(208,0,608,400),(208,208,608,608))]
        for i, crop in enumerate(crops):
            imageid = "test_%.3d" % (4*(original_index-1) + i + 1)
            crop_save_path = os.path.join(foursplit_dir, f"{imageid}.png")
            print(i, original_index, fn, crop_save_path)
            crop.save(crop_save_path)
