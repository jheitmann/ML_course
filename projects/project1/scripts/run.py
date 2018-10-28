
from benchmark_methods import model_output, compute_predictions, logistic_predictions
from proj1_helpers import *
import numpy as np
import csv

test_data_path = 'all/test.csv'
test = load_csv_data(test_data_path)
_, test_data, test_ids, _ = test

weights_path = 'all/weights.npy'
weights = np.load(weights_path)

clean_features_path = 'all/clean_features.npy'
clean_features = np.load(clean_features_path)

parameters_path = 'all/parameters.npy'
parameters = np.load(parameters_path)

out = model_output(test_data, weights, 22, clean_features, parameters)

#TODO how to not manualy have to change? 
#pred = logistic_predictions(out)
pred = compute_predictions(out)

create_csv_submission(test_ids, pred, 'all/ridge_pred.csv')
