from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import numpy as np

X, y = make_regression(n_samples=5390, n_features=28, n_targets=1, random_state=42)
X=X.reshape(5390,28,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1617, random_state=42)

model = Sequential()
model.add(LSTM(28, batch_input_shape=(49,28,1), stateful=True, return_sequences=True))
model.add(LSTM(14, stateful=True))
model.add(Dense(1, activation='relu'))


model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=49,
                    validation_data=(X_test, y_test),
                    verbose=1,
                    shuffle=False)
