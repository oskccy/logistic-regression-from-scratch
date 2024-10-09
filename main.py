import numpy as np
from sigmoid import sigmoid as sig
from dataset import xtrdataset, ytrdataset

alpha = 0.0005
epochs = 10000

vec_m = xtrdataset.shape[1]
vec_n = xtrdataset.shape[0]

W_vector = np.zeros((vec_n, 1))
B_scalar = 0

for i in range(epochs):

    Z = np.dot(W_vector.T, xtrdataset) + B_scalar
    A = sig(Z)

    dW_partialderivative = (1/vec_m) * np.dot(A - ytrdataset, xtrdataset.T)
    dB_partialderivative = (1/vec_m) * np.sum(A - ytrdataset)

    W_vector = W_vector - alpha * dW_partialderivative.T
    B_scalar = B_scalar - alpha * dB_partialderivative


# these values will now fit in a linear algebraic function in sigmoid
print(W_vector, B_scalar)
