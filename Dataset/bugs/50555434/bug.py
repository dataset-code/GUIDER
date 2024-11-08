from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

X, Y = make_regression(1000, 28, n_targets=6, random_state=42)
for row_index in range(0, len(Y)):
    Y[row_index] = Y[row_index] / float(sum(Y[row_index]))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=2)

# create model
model = Sequential()
model.add(Dense(20, input_dim=28, kernel_initializer='normal', activation='relu'))
model.add(Dense(15, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='sigmoid'))

# Compile model
model.compile(optimizer="adam", loss='mae')
# Fit the model
model.fit(X, Y, epochs=2000, verbose=2, validation_split=0.15)
# calculate predictions
# predictions = model.predict(X)
