import cv2
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

PIXEL_DEPTH = 255
IMG_PATCH_SIZE = 16
TEST_IMG_HEIGHT = 608

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
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

# Get prediction for given input image 
def get_prediction(img):
    """
    data = np.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    data_node = tf.constant(data)
    output = tf.nn.softmax(model(data_node))
    output_prediction = s.run(output)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    return img_prediction
    """
    return []

def get_prediction_with_groundtruth(filename, image_idx):
    """
    Get a concatenation of the prediction and groundtruth for given input file
    """

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img)
    cimg = concatenate_images(img, img_prediction)

    return cimg

def get_prediction_with_overlay(filename, image_idx):
    """
    Get prediction overlaid on the original image for given input file
    """

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img)
    oimg = make_img_overlay(img, img_prediction)

    return oimg

def predictions_to_masks(filename, preds, img_height):
    """
    Converts preds into an image mask, and serializes it to filename
    Args:
        filename: filename of serialized mask
        preds: np.array of predictions
        img_height: pixel size of image height        
    """

    num_pred = preds.shape[0]
    # preds[preds >= 0.5] = 1
    # preds[preds < 0.5] = 0
    masks = preds * 255
    masks = np.round(masks).astype('uint8')
    masks = np.squeeze(masks)

    filenames = []
    for i in range(num_pred):
        imageid = f"test_{i+1}"
        image_filename = filename + imageid + ".png"
        
        print ('Predicting ' + image_filename)
        mask = masks[i]
        mask = cv2.resize(mask, dsize=(TEST_IMG_HEIGHT,TEST_IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(image_filename, mask)
        filenames.append(image_filename)
    return filenames

def patch_to_label(patch, foreground_threshold=0.25):
    """
    Assign a label to a patch
    """
    df = np.mean(patch)
    return int(df > foreground_threshold)

def mask_to_submission_strings(image_filename, patch_size=16):
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
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            # Get patch np.array from image np.array
            patch = im[i:i + patch_size, j:j + patch_size]
            # Convert to corresp. label
            label = patch_to_label(patch)
            # Yield resulting string
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

def masks_to_submission(submission_filename, image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))
