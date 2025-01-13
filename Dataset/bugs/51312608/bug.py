import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

VALIDATION_SPLIT=0.1
model = Sequential()
model.add(Dense(2, input_dim=2, kernel_initializer='random_normal', use_bias=True, bias_initializer='random_normal', activation='softmax'))
model.add(Dense(1, use_bias=True, bias_initializer='random_normal', activation='softmax'))

model.summary() # print information about structure of neural net
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01),    
              metrics= ['accuracy']) #mean_squared_error

X_train = np.array([ [0,0],[0,1],[1,0],[1,1] ])
Y_train = np.array([ [0],[1],[1],[0] ])

history = model.fit(X_train, Y_train, 
                batch_size=1, epochs=10, verbose=1, 
validation_split=VALIDATION_SPLIT)
