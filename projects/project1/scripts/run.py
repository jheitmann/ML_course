import csv
import numpy as np
import sys
from proj1_helpers import load_csv_data, create_csv_submission, model_predictions

"""
run.py is used to launch the application of weights on a test dataset and serialize the results.
"""

def load_npy(*npy_paths):
    """
    Returns numpy arrays serialized at npy_paths.
    Args:
        npy_paths : a sequence of serialized np.arrays files paths.
    Returns:
        Deserialized numpy arrays
    """
    return (np.load(p) for p in npy_paths)

# Load the test dataset
_, test_data, test_ids, _ = load_csv_data('all/test.csv')

# Load the weights, feature masks and parameters (mean, std_dev)
weights, clean_features, parameters = load_npy('all/weights.npy', 'all/clean_features.npy', 'all/parameters.npy')

# Runs the weights against the test dataset
pri_jet_num_idx = 22
polynomial_degree = 8
predictions = model_predictions(test_data, weights, pri_jet_num_idx, clean_features, parameters, polynomial_degree)

create_csv_submission(test_ids, predictions, 'all/predictions.csv')
