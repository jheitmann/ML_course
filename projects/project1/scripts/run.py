
from benchmark_methods import model_output, compute_predictions, logistic_predictions
from proj1_helpers import load_csv_data, create_csv_submission
import numpy as np
import csv
import sys

"""
run.py is used to launch the application of weights on a test dataset and serialize the results.

It takes an optional argument 'logistic', which launches logistic_predictions if provided (otherwise, launches compute_predictions)
"""

uses_logistic = True if len(sys.argv) > 1 else False

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
out = model_output(test_data, weights, 22, clean_features, parameters)

# Converts the model output to discrete predictions
pred = (compute_predictions if uses_logistic else logistic_predictions)(out)

# Serializes the predictions to a csv fie
with open('results.csv', 'w', newline='\n', encoding='utf-8') as fp:
    writer = csv.writer(fp)
    for i, id in enumerate(test_ids):
        if not i % 100000: print('Done:', i)
        writer.writerow([id, 1 if pred[i] else -1])

#create_csv_submission(test_ids, pred, 'results.csv') Not cross-compatible
