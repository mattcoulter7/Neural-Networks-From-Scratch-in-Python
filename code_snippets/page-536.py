import matplotlib.pyplot as plt
import numpy as np
import cv2
from zipfile import ZipFile
import urllib
import urllib.request
import os

FOLDER = 'fashion_mnist_images'

np.set_printoptions(linewidth=200)

labels = os.listdir(f'{FOLDER}/train')
print(labels)

files = os.listdir(f'{FOLDER}/train/0')
print(files[:10])
print(len(files))

image_data1 = cv2.imread(f'{FOLDER}/train/7/0002.png', cv2.IMREAD_UNCHANGED)
print(image_data1)

image_data2 = cv2.imread(f'{FOLDER}/train/4/0011.png', cv2.IMREAD_UNCHANGED)
plt.imshow(image_data2)
plt.show()
plt.imshow(image_data2,cmap='gray')
plt.show()