import numpy as np


train = np.load('data/bow_features/train_bow.npy')
val = np.load('data/bow_features/val_bow.npy')
test = np.load('data/bow_features/test_bow.npy')

print(train.shape)
print(val.shape)
print(test.shape)