#Developer: Dillon Pulliam
#Date: 8/21/2020
#Purpose: The purpose of this file is specify the Q-Network architecture


#Imports needed
import keras


#Q-Network class definition
class QNetwork():
	#Name:          __init__
	#Purpose:       define the Keras network architecture
	#Inputs:        environment_name -> name of the environment the model will be used in
	#               lr -> learning rate to use for training
	#Output:        none -> just defines the network architecture
	def __init__(self, environment_name, lr):
		self.model = None
		#Model for the Mountain Car Environment
		if(environment_name == 'MountainCar-v0'):
			self.model = keras.models.Sequential()
			self.model.add(keras.layers.Dense(64, activation='relu', input_shape=(2,)))
			self.model.add(keras.layers.Dense(64, activation='relu'))
			self.model.add(keras.layers.Dense(64, activation='relu'))
			self.model.add(keras.layers.Dense(3, activation='linear'))
			adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
			self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
		#Model for the CartPole Environment
		else:
			self.model = keras.models.Sequential()
			self.model.add(keras.layers.Dense(24, activation='tanh', input_shape=(4,)))
			self.model.add(keras.layers.Dense(24, activation='tanh'))
			self.model.add(keras.layers.Dense(2, activation='linear'))
			adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
			self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
		return

	#Name:          save_model_weights
	#Purpose:       helper function to save Q-Network model parameters
	#Inputs:        none
	#Output:        none -> just saves the model parameters
	def save_model_weights(self):
		self.model.save('model.h5')
		self.model.save_weights('model_weights.h5')
		return

	#Name:          load_model
	#Purpose:       helper function to load a saved Q-Network model
	#Inputs:        none
	#Output:        none -> just loads a saved model
	def load_model(self):
		self.model = keras.models.load_model('model.h5')
		return

	#Name:          load_model_weights
	#Purpose:       helper function to load the weights of a saved Q-Network
	#Inputs:        none
	#Output:        none -> just loads a saved model's parameters
	def load_model_weights(self):
		self.model.load_weights('model_weights.h5')
		return
