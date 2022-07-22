import os
import urllib
import cv2
from Model import Model
import numpy as np
import nnfs
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
nnfs.init()


def load_image(path,imread=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(path, imread)

    return image


# PATH variables
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

# Load the image data
image_data = load_image(askopenfilename(),cv2.IMREAD_GRAYSCALE)

# Transform the Image Data
#image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
image_data = cv2.resize(image_data,(28,28))
image_data = 255 - image_data

# Prepare the Image Data
X_test = np.array([image_data])
y_test = np.array([8])
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
X_test = X_test.reshape(X_test.shape[0], -1)

# Load the model
model = Model.load(MODEL_PATH)

# Determine predictions
model.evaluate(X_test,y_test)
confidences = model.predict(X_test)
predictions = model.output_layer_activation.predictions(confidences)
print(predictions)

# Print the prediction in English
for prediction in predictions:
    print(fashion_mnist_labels[prediction])

# Show the Transformed Image
plt.imshow(image_data, cmap='gray')
plt.show()
