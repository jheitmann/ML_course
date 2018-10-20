
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
        if verbose:print(f'h : {z.shape}, {z[:5]}')
        gradient = np.dot(X.T, (h - y)) / y.size
        if verbose:print(f'grad : {gradient.shape}, {gradient[:5]}')
        weights -= LRATE * gradient
        
        print(f'loss: {loss(h, y)}')

def predict_prob(X):
    return sigmoid(np.dot(X, weights))

def predict(X, threshold=0.5):
    return 1. * (predict_prob(X) >= threshold)

for i in range(19):
    delcoltest=[False]*19
    delcoltest[3]=True
    delcoltest[6]=True
    delcoltest[7]=True
    delcoltest[9]=True
    delcoltest[10]=True
    delcoltest[12]=True
    delcoltest[15]=True
    delcoltest[17]=True
    delcoltest=np.array(delcoltest)
    trainvartest=trainvar[:,delcoltest]
    print(i)
    fit(trainvartest,train[0],verbose=True,max_iters=3)

