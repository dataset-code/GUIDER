from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras.optimizers
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 10, random_state=42, n_classes=2, n_informative=5)
X=X.reshape(1000,10,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40, stratify=y)

in_out_neurons = 1
hidden_neurons = 20

model = Sequential()

# n_prev = 100, 2 values per x axis
model.add(LSTM(hidden_neurons, input_shape=(10, 1)))
model.add(Activation('relu'))
model.add(Dense(in_out_neurons))
model.add(Activation("sigmoid"))
model.add(Activation("softmax"))
rms = keras.optimizers.RMSprop(lr=5, rho=0.9, epsilon=1e-08, decay=0.0)
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.001, nesterov=False)
model.compile(loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=50, validation_data=(X_test, y_test), verbose=1)