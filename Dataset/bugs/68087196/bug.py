import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(12000, 150*150*3, random_state=42, n_classes=120, n_informative=20)
X = X.reshape(12000, 150, 150, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

model = tf.keras.models.Sequential([
    # Old Method #
    tf.keras.layers.Conv2D(8 , (2,2) , activation='LeakyReLU', input_shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(16, (2,2) , activation='LeakyReLU'),
    tf.keras.layers.Conv2D(32, (2,2) , activation='LeakyReLU'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(40, (2,2) , activation='LeakyReLU'),
    tf.keras.layers.Conv2D(56, (2,2) , activation='LeakyReLU'),
    tf.keras.layers.Conv2D(64, (2,2) , activation='LeakyReLU'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(96, (2,2) , activation='LeakyReLU'),
    tf.keras.layers.Conv2D(128, (2,2), activation='LeakyReLU'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='softmax'),
    tf.keras.layers.Dense(120)
    # End Old Method #
    ])

model.compile(optimizer=SGD(lr=0.01),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

history = model.fit(X_train, y_train,epochs=5,validation_data=(X_test, y_test))
