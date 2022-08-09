import csv
import datetime

from pickletools import optimize
from Loss_MeanAbsoluteError import Loss_MeanAbsoluteError
from Loss_MeanSquaredError import Loss_MeanSquaredError
from Loss_BinaryCrossentropy import Loss_BinaryCrossentropy
from Loss_CategoricalCrossentropy import Loss_CategoricalCrossentropy
from Optimizer_Adam import Optimizer_Adam
from Activation_Sigmoid import Activation_Sigmoid
from Activation_Linear import Activation_Linear
from Activation_ReLU import Activation_ReLU
from Layer_Dropout import Layer_Dropout
from Layer_Dense import Layer_Dense
from Model import Model
from nnfs.datasets import sine_data
from Accuracy_Regression import Accuracy_Regression
from Activation_Softmax import Activation_Softmax
from Accuracy_Categorical import Accuracy_Categorical
import numpy as np
import matplotlib.pyplot as plt
import nnfs
nnfs.init()

DATA_PATH = 'Scats Data October 2006.csv'

SCATS_NUMBER_INDEX = 0
GEO_X_INDEX = 3
GEO_Y_INDEX = 4
DATE_INDEX = 9
TIMES_INDEX_START = 10
TIMES_INDEX_END = 106


def get_date(date_string):
    [day, month, year] = date_string.split('/')
    return datetime.date(int(year), int(month), int(day))


def traffic_data():
    X_val = []
    y_val = []
    with open(DATA_PATH, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row_index = -1
        for row in spamreader:
            row_index += 1
            if (row_index < 2):
                continue
            scats_number = row[SCATS_NUMBER_INDEX]
            geo_x = float(row[GEO_X_INDEX])
            geo_y = float(row[GEO_Y_INDEX])
            date = get_date(row[DATE_INDEX])
            day = date.weekday()
            densities = row[TIMES_INDEX_START:TIMES_INDEX_END]

            for i in range(len(densities)):
                density = int(densities[i])
                time = i * 15
                # X_val.append([scats_number,day,time])
                X_val.append([geo_x/100, geo_y/100, (day-3.5) / 3.5, (time - 720) / 720])
                y_val.append([(density - 347.5)/347.5])

    return np.array(X_val), np.array(y_val)

# [
#   [scats_number,day_index,time] || [geo_x,geo_y,day_index,time]
#   [density]
# ]


# generate the data
X, y = traffic_data()

# shuffle the data in preparation for extracting validation data
np.random.shuffle(X)
np.random.shuffle(y)

# extract the validation data
validation_data_size = 20000
X_test = X[0:validation_data_size]
y_test = y[0:validation_data_size]
X = X[validation_data_size:]
y = y[validation_data_size:]

# create the model
model = Model()
model.add(Layer_Dense(X.shape[1], 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-4),
    accuracy=Accuracy_Categorical()
)
model.finalize()
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)

# train the model
model.train(X, y, epochs=100, print_every=10)
