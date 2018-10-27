
import numpy as np

def calculate_mse(e):
    """ Calculate the mse for vector e. """
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """ Calculate the loss. """
    e = y - tx.dot(w)
    return calculate_mse(e)

def least_squares_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.size
    e = y - np.dot(tx,w)
    gradient = (-1/N)*np.dot(tx.T,e)
    return gradient

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

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y,tx,initial_w)]
    w = initial_w
    for n_iter in range(max_iters):
        # Computes gradient
        gradient = least_squares_gradient(y,tx,w)
        # Updates w by gradient and computes loss
        w = w - gamma*gradient
        loss = compute_loss(y,tx,w)
        # Store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
         #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def compute_gradient(y, tx, w): # official solution
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma): # official solution
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y,tx,initial_w)]
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y,tx,batch_size):
            # Computes gradient
            gradient = least_squares_gradient(y_batch,tx_batch,w)
            # Updates w by gradient and computes loss
            w = w - gamma*gradient
            loss = compute_loss(y,tx,w)
            # Store w and loss
            ws.append(w)
            losses.append(loss)
            #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
             #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
    return losses, ws

def least_squares(y, tx):
    """ Least squares regression using normal equations """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations """
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def cross_validation(y, tx, k_indices, k, lambda_):
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
    loss_tr = compute_loss(y_train,tx_train,ws_train)
    loss_te = compute_loss(y_test,tx_test,ws_train)
    return loss_tr, loss_te

def find_optimal_lambda(y,tx):
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
            _, loss_te = cross_validation(y,tx,k_indices,k,lambda_)
            loss_te_sum += loss_te
        testing_losses.append((loss_te_sum/k_fold,lambda_))
        
    optimal_lambda = min(testing_losses)[1]    
    return optimal_lambda

def error(y, tx, w):
    predictions = np.sign(np.dot(tx,w))
    predictions[predictions == 0] = 1
    N = y.size
    return np.count_nonzero(predictions-y)/N

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
    return loss, w # changeme -> nan 

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD """
    threshold = -1 # 1e-8   
    w = initial_w
    losses = [0] # [logistic_loss(y,tx,initial_w)] 

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

def log_cross_validation(y, tx, k_indices, k, lambda_, initial_w, max_iters, gamma):
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

def log_optimal_lambda(y,tx,initial_w, max_iters, gamma):
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
            _, loss_te = log_cross_validation(y,tx,k_indices,k,lambda_,initial_w,max_iters,gamma)
            loss_te_sum += loss_te
        testing_losses.append((loss_te_sum/k_fold,lambda_))
        
    optimal_lambda = min(testing_losses)[1]    
    return optimal_lambda

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD """
    threshold = -1 # 1e-8   
    w = initial_w
    losses = [0] # [logistic_loss(y,tx,initial_w)] 

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