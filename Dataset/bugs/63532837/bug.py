# Importing the essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Getting the dataset
data = pd.read_csv("sales_train.csv")
X = data.iloc[:, 1:-1].values 
y = data.iloc[:, -1].values
# y = np.array(y).reshape(-1, 1)

# Getting the values for november 2013 and 2014 to predict 2015
list_of_november_values = []
list_of_november_values_y = []
for i in range(0, len(y)):
    if X[i, 0] == 10 or X[i, 0] == 22:
        list_of_november_values.append(X[i, 1:])
        list_of_november_values_y.append(y[i])

# Converting list to array
arr_of_november_values = np.array(list_of_november_values)
y_train = np.array(list_of_november_values_y).reshape(-1, 1)

# Scaling the independent values 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(arr_of_november_values)

# Creating the neural network
from keras.models import Sequential
from keras.layers import Dense

nn = Sequential()
nn.add(Dense(units=120, activation='relu'))
nn.add(Dense(units=60, activation='relu'))
nn.add(Dense(units=30, activation='relu'))
nn.add(Dense(units=15, activation='relu'))
nn.add(Dense(units=1, activation='softmax'))
nn.compile(optimizer='adam', loss='mse')
nn.fit(X_train, y_train, batch_size=100, epochs=5)

