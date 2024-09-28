import csv
import math

dataset = []
with open("heart.csv") as datasetfile:
    reader = csv.reader(datasetfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        dataset.append(row)

slice_idx = math.ceil(len(dataset) / 2)

# age -> 0idx, trestbps -> 3idx, chol -> 4idx

trdataset = list(
    map(lambda arr: [arr[0], arr[3], arr[4], arr[13]], dataset[0:slice_idx]))

tdataset = list(
    map(lambda arr: [arr[0], arr[3], arr[4], arr[13]], dataset[(slice_idx+1):-1]))
