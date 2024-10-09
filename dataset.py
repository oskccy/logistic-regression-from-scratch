import csv
import math
import numpy as np

dataset = []
with open("heart.csv") as datasetfile:
    reader = csv.reader(datasetfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        dataset.append(row)

slice_idx = math.ceil(len(dataset) / 2)

xtrdataset = np.array(list(map(lambda arr: np.array(
    [arr[0], arr[3], arr[4]]), dataset[0:slice_idx]))).T

ytrdataset = np.array(
    list(map(lambda arr: arr[13], dataset[0:slice_idx]))).reshape(-1, 1).T

xtdataset = np.array(list(map(lambda arr: np.array(
    [arr[0], arr[3], arr[4]]), dataset[(slice_idx+1):-1]))).T

ytdataset = np.array(
    list(map(lambda arr: arr[13], dataset[(slice_idx+1):-1]))).reshape(-1, 1).T
