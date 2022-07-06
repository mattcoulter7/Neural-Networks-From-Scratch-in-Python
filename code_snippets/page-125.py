import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU
from Activation_Softmax import Activation_Softmax
from Loss_CategoricalCrossentropy import Loss_CategoricalCrossentropy

X, y = spiral_data(samples=100,classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)

print('loss:',loss)