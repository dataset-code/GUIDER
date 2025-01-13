import numpy as np
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
from keras import regularizers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=7, n_targets=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Dense(64, input_shape=(7,), kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
model.add(Dense(1, activation='linear'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=128, shuffle=True)
