from operator import neg
import numpy as np

from Loss import Loss

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    # Backward pass
    def backward(self, dvalues, y_true):
        samples=len(dvalues)
        outputs=len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
