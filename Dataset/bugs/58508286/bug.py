import numpy as np
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=502, n_targets=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

inputs = tf.keras.Input(shape=(502,), name='charge_density_x_max')
hidden1 = tf.keras.layers.Dense(64, activation='sigmoid', name='hidden_1')(inputs)
hidden2 = tf.keras.layers.Dense(64, activation='sigmoid', name='hidden_2')(hidden1)
outputs = tf.keras.layers.Dense(1)(hidden2)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='total_energy')

model.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)