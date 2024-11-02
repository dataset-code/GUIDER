import numpy as np
import tensorflow as tf

prng = np.random.RandomState(1234567891)
x = prng.rand(10000, 1)
y = x

def create_model():
    dropout_nodes = 0.0

    # initialize sequential model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16, input_dim=1, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_nodes))    
    model.add(tf.keras.layers.Dense(8, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout_nodes))   
    model.add(tf.keras.layers.Dense(4, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout_nodes))   
    model.add(tf.keras.layers.Dense(2, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout_nodes))   
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    loss = 'mse'
    metric = ["mae", "mape"]
    opt = tf.keras.optimizers.SGD(learning_rate=1e-2)

    model.compile(loss=loss, optimizer=opt, metrics=[metric])

    return model

model = create_model()

history = model.fit(x=x, y=y,
                    validation_split=0.1, shuffle=False,
                    epochs=20,
                    batch_size=32,
                    verbose=1, )

# pred = model.predict(x)
