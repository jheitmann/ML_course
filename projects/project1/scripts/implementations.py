
import numpy as np

def calculate_mse(e):
    """ Calculate the mse for vector e. """
    return 1/2*np.mean(e**2)

def compute_loss(y, tx, w):
    """ Calculate the loss. """
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_gradient(y, tx, w):
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

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y,tx,initial_w)]
    w = initial_w
    for n_iter in range(max_iters):
        # Computes gradient
        gradient = compute_gradient(y,tx,w)
        # Updates w by gradient and computes loss
        w = w - gamma*gradient
        loss = compute_loss(y,tx,w)
        # Store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def least_square_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y,tx,initial_w)]
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y,tx,batch_size):
            # Computes gradient
            gradient = compute_gradient(y_batch,tx_batch,w)
            # Updates w by gradient and computes loss
            w = w - gamma*gradient
            loss = compute_loss(y,tx,w)
            # Store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    
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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent or SGD """
    pass

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent or SGD """
    pass
