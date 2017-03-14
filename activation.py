import numpy as np
from util import logaddexp

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))

def sigmoid_log(a):
    return -logaddexp(np.zeros(a.shape), -a)