import numpy as np
from util import logaddexp, logsumexp

def sigmoid(a):
    return np.exp(sigmoid_log(a))

def sigmoid_log(a):
    return -logaddexp(np.zeros(a.shape), -a)

def softmax(a):        
    a_exp = np.exp(a - np.max(a, axis=1, keepdims = True))
    return a_exp / np.sum(a_exp, axis=1, keepdims=True)

def softmax_log(a):
    return a - logsumexp(a)