import numpy as np
from tqdm import tqdm
from sigmoid import sigmoid as sig
from dataset import xtrdataset, ytrdataset
from visualizer import loss_visualizer

alpha = 0.00001
epochs = 100000

vec_m = xtrdataset.shape[1]  # this is 513
vec_n = xtrdataset.shape[0]  # this is 3


print(xtrdataset)

# 3x1 shape
W_vector = np.zeros((vec_n, 1))
B_scalar = 0

loss_list = []
epoch_list = []

for i in tqdm(range(epochs), desc="EPOCH ITERATIONS: "):

    Z = np.dot(W_vector.T, xtrdataset) + B_scalar
    A = sig(Z)

    loss = -(1/vec_m) * np.sum(ytrdataset * np.log(A) +
                               (1 - ytrdataset) * np.log(1 - A))

    dW_partialderivative = (1/vec_m) * np.dot(A - ytrdataset, xtrdataset.T)
    dB_partialderivative = (1/vec_m) * np.sum(A - ytrdataset)

    W_vector = W_vector - alpha * dW_partialderivative.T
    B_scalar = B_scalar - alpha * dB_partialderivative

    if i % 2500 == 0:

        loss_list.append(loss)
        epoch_list.append(i)

print(f"TRAINED: {W_vector}")
print(f"TRAINED: {B_scalar}")

loss_visualizer(epoch_list, loss_list)
