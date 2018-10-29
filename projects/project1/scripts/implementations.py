import numpy as np
from proj1_helpers import *

""" ==================== Helper functions ==================== """

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1. / (1. + np.exp(-t))

def split_kth(tx, y, k, k_indices):
    """ Returns test = k-th subgroup of dataset, rest in train. """
    test = [s[k_indices[k]] for s in (y, tx)]
    train_indices = np.array([i for i in range(y.size) if i not in k_indices[k]])
    train = [s[train_indices] for s in (y, tx)]
    # Returns train and test, two tuples of (y, tx)
    return train, test

def _find_optimal_lambda(cross_validation_f, y, tx, *suppl_params):
    """ Computes optimal lambda, with loss_te derived from cross_validation_f """
    seed, k_fold = 1, 4
    lambdas = np.logspace(-4, 0, 30)
    k_indices = build_k_indices(y, k_fold, seed)

    testing_losses = []
    for i, lambda_ in enumerate(lambdas):
        print(f"Iteration {i}")
        loss_te_sum = 0
        for k in range(k_fold):
            _, loss_te = cross_validation_f(y, tx, k_indices, k, lambda_, *suppl_params)
            loss_te_sum += loss_te
        testing_losses.append((loss_te_sum / k_fold, lambda_))
        
    optimal_lambda = min(testing_losses)[1]    
    return optimal_lambda

# Signatures for retrocompatibility
logistic_optimal_lambda = lambda *p: _find_optimal_lambda(logistic_cross_validation, *p)
ridge_optimal_lambda = lambda *p: _find_optimal_lambda(ridge_cross_validation, *p)

""" ==================== Loss functions ==================== """

def calculate_mse(e):
    """ Calculate the mse for vector e. """
    return 1/2*np.mean(e**2)

def least_squares_loss(y, tx, w):
    """ Calculate the loss. """
    e = y - tx.dot(w)
    return calculate_mse(e)

def logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    pred[pred == 0] = 1e-10 # No nan when log(pred) is computed
    pred[pred == 1] = 1 - 1e-10 # No nan when log(1-pred) is computed
    loss = (-y * np.log(pred) - (1. - y) * np.log(1. - pred)).mean()
    return loss

""" ==================== Gradient functions ==================== """

def least_squares_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.size
    e = y - np.dot(tx,w)
    gradient = (-1 / N) * np.dot(tx.T, e)
    return gradient

def logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) / y.size
    return grad

""" ==================== Model training functions ==================== """

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Least squares gradient descent algorithm."""
    w = initial_w
    loss = least_squares_loss(y, tx, initial_w)

    for n_iter in range(max_iters):
        gradient = least_squares_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = least_squares_loss(y, tx, w)
    return loss, w

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Least squares stochastic gradient descent algorithm."""
    w = initial_w
    loss = least_squares_loss(y, tx, initial_w)

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            gradient = least_squares_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient
            loss = least_squares_loss(y, tx, w)
    return loss, w

def least_squares(y, tx):
    """ Least squares regression using normal equations """
    w = np.linalg.solve(*(tx.T.dot(arr) for arr in (tx, y)))
    return least_squares_loss(y, tx, w), w

def learning_by_gradient_descent(y, tx, w, gamma, lambda_=0):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    grad = logistic_gradient(y, tx, w) + 2 * lambda_ * w
    w -= gamma * grad
    loss = logistic_loss(y, tx, w) + lambda_ * w.T.dot(w)
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD """
    threshold, w = 1e-8, initial_w
    losses = [logistic_loss(y, tx, initial_w)]

    for iter in range(max_iters):     
        loss, w = learning_by_gradient_descent(y, tx, w, gamma, lambda_)
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

        # Converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss, w

# Signatures for retrocompatibility
logistic_regression = lambda y, tx, initial_w, max_iters, gamma: reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations """
    aI = lambda_ * np.identity(tx.shape[1])
    a_p, b = (tx.T.dot(arr) for arr in (tx, y))
    return np.linalg.solve(a_p + aI, b)

""" ==================== Cross validation functions ==================== """

def logistic_cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma):
    """return the loss of ridge regression."""
    train, test = split_kth(tx, y, k, k_indices)
    loss_tr, ws_train = reg_logistic_regression(*train, lambda_, initial_w, max_iters, gamma)
    loss_te = logistic_loss(*test, ws_train)
    return loss_tr, loss_te

def ridge_cross_validation(y, tx, k_indices, k, lambda_):
    """ Returns the loss of ridge regression."""
    train, test = split_kth(tx, y, k, k_indices)
    ws_train = ridge_regression(*train, lambda_)
    loss = (least_squares_loss(*s, ws_train) for s in (train, test))
    return loss
