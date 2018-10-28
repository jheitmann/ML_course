import numpy as np
from proj1_helpers import *

def calculate_mse(e):
    """ Calculate the mse for vector e. """
    return 1/2*np.mean(e**2)

def least_squares_loss(y, tx, w):
    """ Calculate the loss. """
    e = y - tx.dot(w)
    return calculate_mse(e)

def least_squares_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.size
    e = y - np.dot(tx,w)
    gradient = (-1/N)*np.dot(tx.T,e)
    return gradient

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Least squares gradient descent algorithm."""
    
    # Define parameters to store w and loss
    w = initial_w
    loss = least_squares_loss(y,tx,initial_w)
    
    for n_iter in range(max_iters):
        
        # Computes gradient
        gradient = least_squares_gradient(y,tx,w)
        
        # Updates w by gradient and computes loss
        w = w - gamma*gradient
        loss = least_squares_loss(y,tx,w)

    return loss, w

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Least squares stochastic gradient descent algorithm."""
    
    # Define parameters to store w and loss
    w = initial_w
    loss = least_squares_loss(y,tx,initial_w)
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y,tx,batch_size):
            
            # Computes gradient
            gradient = least_squares_gradient(y_batch,tx_batch,w)
            
            # Updates w by gradient and computes loss
            w = w - gamma*gradient
            loss = least_squares_loss(y,tx,w)
    
    return loss, w

def least_squares(y, tx):
    """ Least squares regression using normal equations """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = least_squares_loss(y,tx,w)
    return loss, w

def ridge_cross_validation(y, tx, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    tx_test = tx[k_indices[k]]
    y_test = y[k_indices[k]]
    train_indices = np.array([i for i in range(y.size) if i not in k_indices[k]])
    tx_train = tx[train_indices]
    y_train = y[train_indices]
    
    # ridge regression
    ws_train = ridge_regression(y_train,tx_train,lambda_)
    
    # calculate the loss for train and test data
    loss_tr = least_squares_loss(y_train,tx_train,ws_train)
    loss_te = least_squares_loss(y_test,tx_test,ws_train)
    return loss_tr, loss_te

def ridge_optimal_lambda(y,tx):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    testing_losses = []
    
    for i, lambda_ in enumerate(lambdas):
        print("Iteration {}".format(i))
        loss_te_sum = 0
        for k in range(k_fold):
            _, loss_te = ridge_cross_validation(y,tx,k_indices,k,lambda_)
            loss_te_sum += loss_te
        testing_losses.append((loss_te_sum/k_fold,lambda_))
        
    optimal_lambda = min(testing_losses)[1]    
    return optimal_lambda

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations """
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = (-y * np.log(pred) - (1 - y) * np.log(1 - pred)).mean()
    return loss 
    
def logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)/y.size
    return grad

def learning_by_gradient_descent(y, tx, w, gamma,lambda_=0):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    grad = logistic_gradient(y, tx, w)+2*lambda_*w
    w -= gamma * grad
    loss = logistic_loss(y, tx, w) + lambda_ * w.T.dot(w)
    return loss, w 

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD """
    threshold = 1e-8    
    w = initial_w
    losses = [logistic_loss(y,tx,initial_w)]

    # start the logistic regression
    for iter in range(max_iters):
        
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss, w

def logistic_cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    tx_test = tx[k_indices[k]]
    y_test = y[k_indices[k]]
    train_indices = np.array([i for i in range(y.size) if i not in k_indices[k]])
    tx_train = tx[train_indices]
    y_train = y[train_indices]
    
    # regularized logistic regression
    loss_tr, ws_train = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)
    
    # calculate the loss for test data
    loss_te = logistic_loss(y_test,tx_test,ws_train)
    return loss_tr, loss_te

def logistic_optimal_lambda(y,tx,initial_w, max_iters, gamma):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    testing_losses = []
    
    for i, lambda_ in enumerate(lambdas):
        print("Iteration {}".format(i))
        loss_te_sum = 0
        for k in range(k_fold):
            _, loss_te = logistic_cross_validation(y,tx,k_indices,k,lambda_,initial_w,max_iters,gamma)
            loss_te_sum += loss_te
        testing_losses.append((loss_te_sum/k_fold,lambda_))
        
    optimal_lambda = min(testing_losses)[1]    
    return optimal_lambda

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD """
    threshold = 1e-8  
    w = initial_w
    losses = [logistic_loss(y,tx,initial_w)] 

    # start the logistic regression
    for iter in range(max_iters):
        
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma, lambda_)
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss, w