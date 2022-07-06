import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU

X, y = spiral_data(samples=100,classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense1.forward(X)
activation1.forward(dense1.output)

print(activation1.output[:5])