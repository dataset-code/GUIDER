import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 12, random_state=42, n_classes=2, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40, stratify=y)

model = Sequential()

model.add(Dense(32, input_dim=12, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.Accuracy()]
)

model.fit(x=X_train, y=y_train, batch_size=32, epochs=20, verbose=1, validation_split=0.1)
