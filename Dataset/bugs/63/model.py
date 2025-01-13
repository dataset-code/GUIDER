from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, Activation, AveragePooling2D
from keras.models import Model, Sequential
from keras.initializers import glorot_uniform
from keras.models import model_from_json
import keras.backend as K
from keras.optimizers import Adam

# def gaussian(x):
# 	return K.exp(-K.pow(x, 2))

class LeNet(object):
	""" Create LeNet class """
	def __init__(self, config):
		self.config = config
		self.model = None
		self.build_model()

	def build_model(self, input_shape=(32, 32, 1)):
		""" Create model architecture """
		self.model = Sequential()

		# 1st conv layer : CONV + TANH + AVERAGE POOL
		self.model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), 
						kernel_initializer=glorot_uniform(seed=0), 
						padding="valid", 
						input_shape=input_shape, 
						activation='tanh',
						name="Conv_1"))
		self.model.add(AveragePooling2D(pool_size=(2, 2), name="AvgPool_1"))

		# 2nd conv layer : CONV + TANH + AVERAGE POOL
		self.model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), 
						kernel_initializer=glorot_uniform(seed=0), 
						padding="valid", 
						activation='tanh',
						name="Conv_2"))
		self.model.add(AveragePooling2D(pool_size=(2, 2), name="AvgPool_2"))

		# 3rd conv layer : CONV + TANH
		self.model.add(Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), 
						kernel_initializer=glorot_uniform(seed=0), 
						padding="valid",
						activation='tanh', 
						name="Conv_3"))
		# Flatten
		self.model.add(Flatten(name="Flatten"))

		# Fully connected layer 1 : DENSE + TANH
		self.model.add(Dense(84, activation='tanh',name="FC_1"))

		# Output with softmax
		self.model.add(Dense(10, activation='softmax',name="FC_2"))

		# Compile model
		self.compile()

	def compile(self):
		""" 
		Compile model using Adam optimizer
		"""
		self.model.compile(optimizer=
			Adam(lr=self.config["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-7), 
			loss='categorical_crossentropy', 
			metrics=['accuracy'])		

	# def save_model(self):
	# 	""" Save network model """

	# 	if self.model is None:
	# 		raise Exception("You have to build the model first")

	# 	# Serialize model to JSON
	# 	model_json = self.model.to_json()
	# 	with open("./saved/model.json", "w") as json_file:
	# 		json_file.write(model_json)

	# 	print("Model network saved")

	def save_weights(self):
		""" Save network weights """
		if self.model is None:
			raise Exception("You have to build the model first")
		# Serialize weights to hdf5
		self.model.save_weights("saved/model.hdf5")

		print("Model weights saved")

		# def load_model(self):
		# 	""" Load model architecture """
			
		# 	with open("./saved/model.json", "r") as json_file:
		# 		loaded_model_json = json_file.read()

		# 	loaded_model = model_from_json(loaded_model_json)

		# 	return loaded_model 


	def load_weights(self, checkpoint_path):
		""" Load model weights """
		if self.model is None:
			raise Exception("You have to build the model first")		

		print("Loading model checkpoint %s ... \n" % (checkpoint_path))
		self.model.load_weights(checkpoint_path)
		print("Model weights loaded")

