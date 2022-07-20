import os
import urllib.request
import urllib
from zipfile import ZipFile
import cv2
from pickletools import optimize
from Loss_MeanAbsoluteError import Loss_MeanAbsoluteError
from Loss_MeanSquaredError import Loss_MeanSquaredError
from Loss_BinaryCrossentropy import Loss_BinaryCrossentropy
from Loss_CategoricalCrossentropy import Loss_CategoricalCrossentropy
from Optimizer_Adam import Optimizer_Adam
from Activation_Softmax import Activation_Softmax
from Activation_Sigmoid import Activation_Sigmoid
from Activation_Linear import Activation_Linear
from Activation_ReLU import Activation_ReLU
from Layer_Dropout import Layer_Dropout
from Layer_Dense import Layer_Dense
from Model import Model
from nnfs.datasets import spiral_data
from Accuracy_Regression import Accuracy_Regression
from Accuracy_Categorical import Accuracy_Categorical
import numpy as np
import matplotlib.pyplot as plt
import nnfs

nnfs.init()

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

model = Model()
model.add(Layer_Dense(X.shape[1],128,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(128,128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128,10))
model.add(Activation_Softmax())
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001,decay=1e-3),
    accuracy=Accuracy_Categorical()
)
model.finalize()
model.train(X,y,validation_data=(X_test, y_test),epochs=10,batch_size=128,print_every=100)
model.evaluate(X_test,y_test)