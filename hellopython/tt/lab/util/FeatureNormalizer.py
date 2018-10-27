import numpy as np

def featureNormalizeV(X):
    return __featureNormalize(X,0)

def featureNormalizeH(X):
    return __featureNormalize(X,1)

def __featureNormalize(X, axis):
    mu = np.mean(X,axis=axis,keepdims=True);
    sigma = np.std(X,axis=axis,keepdims=True);
    X_norm = (X-mu) / sigma;
    return (mu, sigma, X_norm)