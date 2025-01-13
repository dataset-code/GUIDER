from keras.layers import Dense, LSTM
from keras.models import Sequential
import numpy as np
from keras.utils import to_categorical

train_x = np.array([np.random.rand(1, 1000)[0] for i in range(10000)])
train_y = (np.random.randint(1,150,10000))
sample = np.random.rand(1,1000)[0]

train_x = train_x.reshape(10000,1000,1)
# train_y = to_categorical(train_y, num_classes=150)

model = Sequential()
model.add(LSTM(32, input_shape = (1000,1)))
model.add(Dense(150, activation='relu'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
history = model.fit(train_x, train_y,
            batch_size=128, epochs=1,
            verbose = 1)
