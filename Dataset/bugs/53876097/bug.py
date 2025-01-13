from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Sequential
import numpy as np

X, y = make_classification(1000, 64, random_state=42, n_classes=2, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40, stratify=y)


model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(64,)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=5)
