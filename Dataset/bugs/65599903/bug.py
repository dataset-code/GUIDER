from keras.layers import Input, Dense, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.models import Model, Sequential
import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(1000, 10, random_state=42, n_classes=2, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40, stratify=y)

model = Sequential([
      Dense(64, activation='relu'),
      Dropout(0.2),
      Dense(1, activation='softmax')
  ])

model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics='accuracy')

history = model.fit(X_train, y_train,epochs=30)
