from keras.layers import Input, Dense, Flatten, LSTM, Bidirectional
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# X, y = make_classification(10000, 4096, random_state=42, n_classes=39, n_informative=100)
# X = X.reshape(X.shape[0], 4096, 1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

df = pd.read_csv("labeled_frames.csv")
labelencoder = LabelEncoder()
df['Phoneme'] = labelencoder.fit_transform(df['Phoneme'])
labels = np.asarray(df[['Phoneme']].copy())
df = df.drop(df.columns[0], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(df, labels, random_state = 42, test_size = 0.2, stratify = labels)
X_train = tf.reshape(X_train, (8113, 4096, 1))
X_test = tf.reshape(X_test, (2029, 4096, 1))

model = Sequential()
model.add(Conv1D(filters= 128, kernel_size=3, activation ='relu',strides = 2, padding = 'valid', input_shape= (4096, 1)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters= 128, kernel_size=3, activation ='relu',strides = 2, padding = 'valid'))
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters= 128, kernel_size=3, activation ='relu',strides = 2, padding = 'valid'))
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters= 128, kernel_size=3, activation ='relu',strides = 2, padding = 'valid'))
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(39, activation ='softmax')) 

optimizer = keras.optimizers.Adam(lr=0.4)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train,y_train, epochs = 50, batch_size = 2048, validation_data = (X_test, y_test), shuffle = True)
