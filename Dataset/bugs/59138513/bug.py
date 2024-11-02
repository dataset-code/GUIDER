from keras.layers import Dense
from keras.models  import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 12, random_state=42, n_classes=2, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40, stratify=y)

model = Sequential()
model.add(Dense(9))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1, activation="softmax"))

model.compile(loss = "binary_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_data = (X_test, y_test))
