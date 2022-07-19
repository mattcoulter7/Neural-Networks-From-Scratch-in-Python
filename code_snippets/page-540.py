import matplotlib.pyplot as plt
import numpy as np
import cv2
from zipfile import ZipFile
import urllib
import urllib.request
import os

FOLDER = 'fashion_mnist_images'


def load_mnist_dataset(dataset, path):
    X = []
    y = []

    for label in os.listdir(os.path.join(path, dataset)):
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(
                path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_mnist_data(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test

# load the data
X, y, X_test, y_test = create_mnist_data(FOLDER)

# scale values between -1 and 1
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

print(X.min(), X.max())
print(X.shape)

# flatten
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# shuffle the data so it is no organised
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]