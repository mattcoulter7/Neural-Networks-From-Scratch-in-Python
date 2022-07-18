from pickletools import optimize
from Loss_MeanAbsoluteError import Loss_MeanAbsoluteError
from Loss_MeanSquaredError import Loss_MeanSquaredError
from Loss_BinaryCrossentropy import Loss_BinaryCrossentropy
from Optimizer_Adam import Optimizer_Adam
from Activation_Sigmoid import Activation_Sigmoid
from Activation_Linear import Activation_Linear
from Activation_ReLU import Activation_ReLU
from Layer_Dropout import Layer_Dropout
from Layer_Dense import Layer_Dense
from Model import Model
from nnfs.datasets import sine_data
from Accuracy_Regression import Accuracy_Regression
import numpy as np
import matplotlib.pyplot as plt
import nnfs
nnfs.init()


X, y = sine_data()



model = Model()

model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    accuracy=Accuracy_Regression()
)
model.finalize()

#init graph stuff
plt.ion()
fig = plt.figure()
ax = fig.add_subplot()
line1, = ax.plot(X, y, '-r')
line2, = ax.plot(X, np.zeros_like(X), '-b')
def update_graph(output):
    line2.set_ydata(output)
    fig.canvas.draw()
    fig.canvas.flush_events()

# train the model
model.train(X, y, epochs=10000, print_every=10,callback = update_graph)
