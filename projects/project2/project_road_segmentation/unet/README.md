
Machine Learning Project 2
=====

Julien Niklas Heitmann
Philipp Khlebnikov
Louis Gabriel Jean Landelle

##Environement preparation

In order to reproduce our results, you have to perform those preparation steps, in some ROOT_FOLDER:

1) Unzip the datasets in ROOT_FOLDER (in order to have test_set_images/ and training/)

2) Place all the .py scripts in ROOT_FOLDER

3) Launch utrain.py to retrain our model on the training dataset. The script transforms the environement in this fashion:
    - the following empty folders are generated:
        ROOT_FOLDER/data/train/image
        ROOT_FOLDER/data/train/label
        ROOT_FOLDER/data/test/image
        ROOT_FOLDER/data/test/foursplit
        ROOT_FOLDER/results/train/label
        ROOT_FOLDER/results/train/logits
        ROOT_FOLDER/results/train/overlay
        ROOT_FOLDER/results/test/label
        ROOT_FOLDER/results/test/logits
        ROOT_FOLDER/results/test/overlay
    - the training/ images are transfered to data/train/image/, groundtruth to data/train/label/
    - the training script runs, and generates a *.hdf5 checkpoint in results/, named to represent the chosen parameters

4) Launch utest.py on this checkpoint to generate the csv submission along with intermediate visualizations
    - test_set_images/test_*/*.png are transfered to data/test/image/
    - the testing script runs, and generates images in results/test/label, results/test/logits, results/test/overlay
    - it also generates a submission csv in results/

## Models
