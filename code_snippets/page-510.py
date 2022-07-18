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


X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

model = Model()

model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05,decay=5e-5),
    accuracy=Accuracy_Categorical()
)
model.finalize()

# train the model
model.train(X, y, validation_data=(X_test,y_test), epochs=10000, print_every=100)