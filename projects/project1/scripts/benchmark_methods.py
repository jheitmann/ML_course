
import numpy as np
from proj1_helpers import *
from implementations import *

def extend_and_standardize(input_data, mean=None, std=None):
    if mean is not None and std is not None:
        mean_x = mean
        std_x = std
        tx = (input_data - mean) / std
        num_samples = input_data.shape[0]
        tx = np.c_[np.ones(num_samples), tx]
    else: 
        x, mean_x, std_x = standardize(input_data)
        tx = build_model_data(x)
    return tx, mean_x, std_x

def compute_predictions(model_output):
    predictions = model_output
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    return predictions

def logistic_predictions(model_output):
    predictions = np.sign(model_output)
    predictions[predictions == -1] = 0
    return predictions

def compute_accuracy(y, predictions):
    N = y.size
    accuracy = 1 - (np.count_nonzero(predictions-y)/N)
    print("Accuracy: {}".format(accuracy))

def model_output(tx, ws, pri_jet_num_idx, clean_features, parameters):
    cond_null = tx[:, pri_jet_num_idx] == 0
    cond_one = tx[:, pri_jet_num_idx] == 1
    cond_plural = tx[:, pri_jet_num_idx] >= 2
    conditions = (cond_null, cond_one, cond_plural)

    N = tx.shape[0]
    model_output = np.zeros(N)
    for pri_jet_num, cond in enumerate(conditions):
        select_features = clean_features[pri_jet_num]
        reduced_dset = tx[cond][:,select_features]
        mean, std = parameters[pri_jet_num]
        extended_dset,_,_ = extend_and_standardize(reduced_dset,mean,std)
        weight = ws[pri_jet_num]
        sub_output = extended_dset.dot(weight)
        model_output[cond] = sub_output
        
    return model_output
