from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import time

X, y = make_classification(1000, 82, random_state=42, n_classes=2, n_informative=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
print(y_test)

model = Sequential([
    Dense(256, activation='tanh', input_shape=(82,)),
    Dense(2, input_shape=(256,), activation='tanh'),
    Dense(1, activation='softmax')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=X_train, y=y_train, shuffle=True, epochs=5)

print(model.predict(X_test))