import os
import urllib
import cv2
from Model import Model
import numpy as np
import nnfs

nnfs.init()


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

# PATH variables
FOLDER = 'fashion_mnist_images'
MODEL_PATH = 'fashion_mnist.model'

# Classification Labels
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# load the data
X, y, X_test, y_test = create_mnist_data(FOLDER)
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
X_test = X_test.reshape(X_test.shape[0], -1)

# load the model
model = Model.load(MODEL_PATH)

# determine predictions
confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)
print(predictions)
print(y_test[:5])

# map predictions to labels
for prediction in predictions:
    print(fashion_mnist_labels[prediction])
