# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, step=None):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    features = np.genfromtxt(data_path, delimiter=",", dtype=str, max_rows=1)
    features = features[2:]
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if step:
        yb = yb[::step]
        input_data = input_data[::step]
        ids = ids[::step]

    return yb, input_data, ids, features

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred

def model_predictions(tx, ws, pri_jet_num_idx, clean_features, parameters):
    cond_null = tx[:, pri_jet_num_idx] == 0
    cond_one = tx[:, pri_jet_num_idx] == 1
    cond_plural = tx[:, pri_jet_num_idx] >= 2
    conditions = (cond_null, cond_one, cond_plural)

    N = tx.shape[0]
    model_predictions = np.zeros(N)
    for pri_jet_num, cond in enumerate(conditions):
        weight = ws[pri_jet_num]
        select_features = clean_features[pri_jet_num]
        reduced_dset = tx[cond][:,select_features]
        poly_dset = build_poly(reduced_dset,3)
        mean, std = parameters[pri_jet_num]
        extended_dset, _, _ = extend_and_standardize(poly_dset[:,1:],mean,std)
        sub_prediction = predict_labels(weight,extended_dset)
        model_predictions[cond] = sub_prediction
        
    return model_predictions

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def build_model_data(x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = x.shape[0]
    tx = np.c_[np.ones(num_samples), x]
    return tx

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

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

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset. """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
