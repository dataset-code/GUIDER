import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

X_daten = [
    [-4],
    [-3],
    [-2],
    [-1],
    [ 0],
    [ 1],
    [ 2],
    [ 3],
    [ 4]
]
Y_daten = X_daten.copy()

test_anzahl = 2

X_train = np.array(X_daten[:-test_anzahl])
Y_train = np.array(Y_daten[:-test_anzahl])
X_test  = np.array(X_daten[-test_anzahl:])
Y_test  = np.array(Y_daten[-test_anzahl:])

print("1 X_train ", X_train.shape)
print("1 Y_train ", Y_train.shape)
print("1 X_test  ", X_test.shape)
print("1 Y_test  ", Y_test.shape)
print("-"*20)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1, Y_train.shape[1]))
X_test  = np.reshape(X_test , ( X_test.shape[0], 1,  X_test.shape[1]))
Y_test  = np.reshape(Y_test , ( Y_test.shape[0], 1,  Y_test.shape[1]))

print("2 X_train ", X_train.shape)
print("2 Y_train ", Y_train.shape)
print("2 X_test  ", X_test.shape)
print("2 Y_test  ", Y_test.shape)
print("-"*20)

# Neural Netzwerk
neuronen      = 100
layer         = 2
batch_size    = 10
epoch         = 10
input_anzahl  = 1
output_anzahl = 1
#dropout       = 0.3
activation    = "sigmoid"
optimizer     = "Adam"

model = Sequential()

model.add(Dense(neuronen, input_shape=(input_anzahl,), activation=activation))

for _ in range(layer):
    model.add(Dense(neuronen, activation=activation))
    #model.add(Dropout(dropout))

model.add(Dense(output_anzahl, activation=activation)) # Output

model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])

# Training
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch, verbose=1, shuffle=True, validation_data=(X_test, Y_test))
