from keras.layers import Dense
from keras.models import Sequential
import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 29, random_state=42, n_classes=2, n_informative=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40, stratify=y)

model = Sequential()
model.add(Dense(29, input_shape=(29,)))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(2))

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=32)
