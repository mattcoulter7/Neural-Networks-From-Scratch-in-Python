import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Optimizer_Adagrad import Optimizer_Adagrad
nnfs.init()

from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU
from Activation_Softmax_Loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy

X, y = spiral_data(samples=100,classes=3)

dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adagrad(decay=1e-4)

for i in range(10001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if (len(y.shape) == 2):
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not i % 100:
        print(  f'epoch:{i}, ' + 
                f'acc:{accuracy:.3f}, ' + 
                f'loss:{loss:.3f}, ' + 
                f'lr:{optimizer.current_learning_rate:.3f}, '
            )

    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()