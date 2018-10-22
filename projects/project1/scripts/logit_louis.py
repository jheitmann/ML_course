
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def fit(X, y, w_0=None, max_iters=None, l_rate=None, verbose=False):
    weights = w_0 or np.zeros(X.shape[1])
    ITERATIONS = max_iters or 500
    LRATE = l_rate or 0.03

    for i in range(ITERATIONS):
        z = np.dot(X, weights)
        if verbose:print(f'z : {z.shape}, {z[:5]}')
        h = sigmoid(z)
        if verbose:print(f'h : {h.shape}, {h[:5]}')
        gradient = np.dot(X.T, (h - y)) / y.size
        if verbose:print(f'grad : {gradient.shape}, {gradient[:5]}')
        weights -= LRATE * gradient
        
        print(f'loss: {loss(h, y)}')
        return weights

def predict_prob(X,y):
    return sigmoid(np.dot(X, fit(X,y)))

def predict(X, y, threshold=0.5):
    return 1. * (predict_prob(X,y) >= threshold)

pred = predict_prob(trainvartes,train[0])

result=sum(1. * (pred!=train[0]))/len(train[0])

