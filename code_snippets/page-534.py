from zipfile import ZipFile
import urllib
import urllib.request
import os

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'
if not os.path.isfile(FILE):
    print(f'downloading {URL} and saving as {FILE}')
    urllib.request.urlretrieve(URL,FILE)

print('unzipping images')
with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

labels = os.listdir(f'{FOLDER}/train')
print(labels)