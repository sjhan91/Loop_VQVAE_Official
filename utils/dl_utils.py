import numpy as np


def inverse_sigmoid(epoch, k=12):
    return k / (k + np.exp(epoch/k))


def kl_annealing(epoch, start, end, rate=0.9):
    return end + (start - end)*(rate)**epoch
