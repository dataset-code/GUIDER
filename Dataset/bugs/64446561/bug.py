from keras.layers import Dense
from keras import Sequential

import numpy as np

x = np.array([[0., 0.],
              [1., 1.],
              [1., 0.],
              [0., 1.]], dtype=np.float32)

y = np.array([[0.], 
              [0.], 
              [1.], 
              [1.]], dtype=np.float32)

model = Sequential()
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='SGD', loss='mean_squared_error', metrics='accuracy')
model.fit(x, y, batch_size=1, epochs=1000, verbose=False)

# pred = model.predict_on_batch(x)
# print(pred)
