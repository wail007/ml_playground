import numpy as np

def logaddexp(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))

def logsumexp(x):
    m = np.max(x, axis=1, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))