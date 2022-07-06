import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
from Layer_Dense import Layer_Dense

X, y = spiral_data(samples=100,classes=3)

dense1 = Layer_Dense(2,3)

dense1.forward(X)

print(dense1.output[:5])