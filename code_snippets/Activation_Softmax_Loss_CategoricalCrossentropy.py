import numpy as np

from Activation_Softmax import Activation_Softmax
from Loss_CategoricalCrossentropy import Loss_CategoricalCrossentropy

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -= 1
        self.dinputs = self.dinputs / samples