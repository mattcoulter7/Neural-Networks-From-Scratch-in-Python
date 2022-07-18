import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Optimizer_Adam import Optimizer_Adam
nnfs.init()

from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU
from Activation_Softmax_Loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy

X, y = spiral_data(samples=1000,classes=3)

dense1 = Layer_Dense(2,100,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(100,200)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(200,100)
activation3 = Activation_ReLU()
dense4 = Layer_Dense(100,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate=0.072,decay=5e-3)

for i in range(10000):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    dense4.forward(activation3.output)

    data_loss = loss_activation.forward(dense4.output,y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2) + loss_activation.loss.regularization_loss(dense3) + loss_activation.loss.regularization_loss(dense4)
    loss = data_loss + regularization_loss
    
    predictions = np.argmax(loss_activation.output, axis=1)
    if (len(y.shape) == 2):
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not i % 100:
        print(  f'epoch:{i}, ' + 
                f'acc:{accuracy:.3f}, ' + 
                f'data_loss:{data_loss:.3f}, ' + 
                f'reg_loss:{regularization_loss:.3f}, ' + 
                f'loss:{loss:.3f}, ' + 
                f'lr:{optimizer.current_learning_rate:.3f}, '
            )

    loss_activation.backward(loss_activation.output,y)
    dense4.backward(loss_activation.dinputs)
    activation3.backward(dense4.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.update_params(dense4)
    optimizer.post_update_params()


X_test, y_test = spiral_data(samples=1000,classes=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
dense4.forward(activation3.output)
loss = loss_activation.forward(dense4.output,y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if (len(y.shape) == 2):
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')