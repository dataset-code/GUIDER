from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import keras

X, y = make_classification(1000, 400, random_state=42, n_classes=2, n_informative=5)
X = X.reshape(1000,400,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40, stratify=y)

model = Sequential()
model.add(LSTM(128, input_shape=(400,1), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(lr=0.01)

# mean_squared_error = mse
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=3, validation_split = 0.1, shuffle=True, batch_size = 64)
