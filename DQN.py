#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import collections
import random
import matplotlib.pyplot as plt
import os
import time

class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, environment_name, lr):
		# Define your network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
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
			#Currenly running all relu with first layer being 48 units and softmax for last layer
			self.model = keras.models.Sequential()
			self.model.add(keras.layers.Dense(24, activation='tanh', input_shape=(4,)))
			self.model.add(keras.layers.Dense(24, activation='tanh'))
			self.model.add(keras.layers.Dense(2, activation='linear'))
			adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
			self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights.
		self.model.save('model.h5')
		self.model.save_weights('model_weights.h5')
		return

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		self.model = keras.models.load_model('model.h5')
		#self.model = keras.models.load_model(model_file)
		return

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights.
		# e.g.: self.model.load_state_dict(torch.load(model_file))
		self.model.load_weights('model_weights.h5')
		#self.model.load_weights(weight_file)
		return


class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.

		# Hint: you might find this useful:
		# 		collections.deque(maxlen=memory_size)
		self.replay_memory = collections.deque(maxlen=memory_size)
		self.burn_in = burn_in

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
		# You will feed this to your model to train.
		r_batch = random.sample(self.replay_memory, batch_size)
		return np.array(r_batch)

	def append(self, transition):
		# Appends transition to the memory.
		self.replay_memory.append(transition)
		return


class DQN_Agent():

	# In this class, we will implement functions to do the following.
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy.
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.

		#Create the environment
		self.environment_name = environment_name
		self.env = gym.make(self.environment_name)

		#Variables to store the total number of episodes, discount factor (gamma),
		#batch size, epsilon, epsilon minimum value, epsilon decay value, and learning rate
		self.episodes = None
		self.gamma = None
		self.batch_size = None
		self.epsilon = None
		self.learning_rate = None
		#For the Mountain-Car environment
		if(self.environment_name == "MountainCar-v0"):
			self.episodes = 3001
			self.gamma = 0.99
			self.batch_size = 32
			self.epsilon = 1
			self.epsilon_min = 0.01
			self.epsilon_decay = (self.epsilon-self.epsilon_min)/2000
			self.learning_rate = 0.001
		#For the CartPole Environment
		else:
			self.episodes = 5001
			self.gamma = 0.99
			self.batch_size = 32
			self.epsilon = 1
			self.epsilon_min = 0.01
			self.epsilon_decay = (self.epsilon-self.epsilon_min)/self.episodes
			self.learning_rate = 0.001

		#Variables to use for plotting while training
		self.train_episodes_per_test = 100
		self.test_episodes = 50

		#Variables to use for the number of video captures
		self.total_videos = 4-1
		self.video_interval = int(self.episodes / self.total_videos)

		#Size of the replay buffer and burn in value
		self.memory_buffer_size = 50000
		self.burn_in_value = 10000

		#Create the network itself as well as the replay memory data structure
		self.DQN_Model = QNetwork(self.environment_name, self.learning_rate)
		self.Memory = Replay_Memory(self.memory_buffer_size, self.burn_in_value)

		#Variabes to store the best model and best error
		self.DQN_Best = None
		self.best_reward = None
		self.best_td_error = None
		if(self.environment_name == "MountainCar-v0"):
			self.best_reward = -1000

	def epsilon_greedy_policy(self, q_values):
		#Determine the number of actions to get and compute random actions and our greedy actions
		batch_size = len(q_values)
		random_action = np.random.randint(0, len(q_values[0]), size=batch_size)
		greedy_action = np.argmax(q_values, axis=1)

		#Create a vector to store all true actions and fill it with random or greedy actions based on epsilon
		actions = np.zeros(batch_size, dtype=int)
		for i in range(batch_size):
			value = random.uniform(0, 1)
			if(value > self.epsilon):
				actions[i] = greedy_action[i]
			else:
				actions[i] = random_action[i]
		return actions

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		action = np.argmax(q_values, axis=1)
		return action

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.

		# When use replay memory, you should interact with environment here, and store these
		# transitions to memory, while also updating your model.

		#First we need to execute the 'burn in' to fill the memory buffer
		start_time = time.time()
		self.burn_in_memory()

		#Variables to store the outputs of policy progression through the environment
		state = None
		action = None
		reward = None
		next_state = None
		done = None
		not_termination = None

		#Numpy array to store test rewards and TD error values
		average_rewards = np.zeros((int(self.episodes/self.train_episodes_per_test)+1,2), dtype=float)
		average_td_error = np.zeros((int(self.episodes/self.train_episodes_per_test)+1,2), dtype=float)

		for i in range(self.episodes):
			state = self.env.reset()
			done = False
			while(done == False):
				#Get the q-values using our policy and get an action
				q_values = self.DQN_Model.model.predict(np.array([state]), batch_size=1)
				action = self.epsilon_greedy_policy(q_values)
				#Take our action to move to a next state, get a reward, and determine if our environment has terminated
				next_state, reward, done, _ = self.env.step(action[0])
				#Set the termination variable as it will be helpful for calculating y_j
				if(done == False):
					not_termination = 1.0
				else:
					not_termination = 0.0
				#Create the tuple to store to the replay memory and append it
				tuple = [state, q_values, np.array(reward), next_state, np.array(not_termination)]
				self.Memory.append(tuple)

				#Sample a batch from the memory buffer
				batch = self.Memory.sample_batch(self.batch_size)

				#Get the true_y values to use for updating the model repacing the target action with (r + gamma * max_a' Q(...) )
				true_y = self.DQN_Model.model.predict(np.vstack(batch[:, 0]), batch_size=self.batch_size)
				#Compute the new y values to use for the update action and get the action indices corresping to these updates
				update_y = batch[:,2] + (batch[:,4] * self.gamma * np.max(self.DQN_Model.model.predict(np.vstack(batch[:,3]), batch_size=self.batch_size), axis=1))
				update_indices = np.argmax(np.vstack(batch[:,1]), axis=1)
				#Replace the new y values for performed action in true_y
				for count in range(self.batch_size):
					true_y[count,update_indices[count]] = update_y[count]

				#Train the model over a batch
				history = self.DQN_Model.model.fit(x=np.vstack(batch[:, 0]), y=true_y, epochs=1, batch_size=self.batch_size, verbose=0)
				"""loss = history.history['loss'][-1]
				acc = history.history['acc'][-1]
				print('loss= %.3f; accuracy = %.1f%%' % (loss, 100 * acc))"""
				#Set the current state to the next state from taking the action for the next loop
				state = next_state

			#Decay the value of epsilon
			self.epsilon -= self.epsilon_decay
			#Evaluate the model by testing it every train_episodes_per_test iterations
			if(i % self.train_episodes_per_test == 0):
				print("Episode: ", i)
				test_reward, td_error = self.test()
				#Store the specific episode we are testing on and the test reward itself and TD error
				average_rewards[int(i/self.train_episodes_per_test),0] = i
				average_td_error[int(i/self.train_episodes_per_test),0] = i
				average_rewards[int(i/self.train_episodes_per_test),1] = test_reward
				average_td_error[int(i/self.train_episodes_per_test),1] = td_error

			#Capture a video of the agent in the environment at a specific time as it learns
			if(i % self.video_interval == 0):
				self.test_video(i)

		#Get the total training time
		end_time = time.time()
		print("Total Runtime:", end_time - start_time, "seconds")
		#Plot the rewards throughout training and td error through training
		self.plot_graphics(average_rewards, average_td_error)
		return

	# Note: if you have problems creating video captures on servers without GUI,
	#       you could save and relaod model to create videos on your laptop.
	def test_video(self, epi):
		# Usage:
		# 	you can pass the arguments within agent.train() as:
		# 		if episode % int(self.num_episodes/3) == 0:
		#       	test_video(self, self.environment_name, episode)
		new_environment = gym.make(self.environment_name)
		save_path = "./videos-%s-%s" % (self.environment_name, epi)
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		# To create video
		env = gym.wrappers.Monitor(new_environment, save_path, force=True)
		reward_total = []
		state = new_environment.reset()
		done = False
		while not done:
			new_environment.render()
			q_values = self.DQN_Model.model.predict(np.array([state]), batch_size=1)
			action = self.greedy_policy(q_values)
			next_state, reward, done, info = new_environment.step(action[0])
			state = next_state
			reward_total.append(reward)
		#print("reward_total: {}".format(np.sum(reward_total)))
		env.close()
		new_environment.close()

	def plot_graphics(self, average_rewards, average_td_error):
		#Here we plot the average cumulative test reward over training episodes collected
		#at specific intervals during training
		plt.plot(average_rewards[:,0], average_rewards[:,1], color='blue')
		plt.ylabel("Average Cumulative Test Reward")
		plt.xlabel("Total Training Episodes")
		plt.title('Average Cumulative Test Reward vs. Training Episodes')
		plt.show()

		#Here we plot the average TD error over training episodes collected
		#at specific intervals during training
		plt.plot(average_td_error[:,0], average_td_error[:,1], color='red')
		plt.hlines(0, xmin=average_td_error[0,0], xmax=average_td_error[len(average_td_error)-1,0], color='grey')
		plt.ylabel("Average TD Error")
		plt.xlabel("Total Training Episodes")
		plt.title('Average TD Error vs. Training Episodes')
		plt.show()
		return

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.

		#Variables to store the outputs of policy progression through the environment
		state = None
		action = None
		reward = None
		next_state = None
		done = None
		not_termination = None

		#Numpy array to store rewards over testing per episode
		all_rewards = np.zeros(self.test_episodes, dtype=int)

		#Holds the TD Error sum and a counter for total steps
		steps = 0
		td_error_sum = 0

		for i in range(self.test_episodes):
			state = self.env.reset()
			done = False
			while(done == False):
				#Get the q-values using our policy and get an action
				q_values = self.DQN_Model.model.predict(np.array([state]), batch_size=1)
				action = self.greedy_policy(q_values)
				#Take our action to move to a next state, get a reward, and determine if our environment has terminated
				next_state, reward, done, _ = self.env.step(action[0])
				#Add the current reward to the data structure storing the rewards
				all_rewards[i] += reward
				#Add to the td_error_sum variable as well as the steps we have computed it over
				td_error_sum += abs((reward + (self.gamma * np.max(self.DQN_Model.model.predict(np.array([next_state]), batch_size=1)))) - np.max(q_values))
				steps += 1
				#Set the current state to the next state from taking the action for the next loop
				state = next_state
		#Compute the cumulative reward
		average_reward = np.mean(all_rewards)

		#To prevent the over-fitting problem of the MountainCar Environment
		if(self.environment_name == "MountainCar-v0"):
			#Only save the model if it performs better
			#If the model fails to perform better re-load the best model, half
			#the learning rate, and re-compile the model
			if(average_reward >= self.best_reward):
				self.DQN_Best = copy.deepcopy(self.DQN_Model)
				self.best_reward = average_reward
				self.best_td_error = td_error_sum/steps
			else:
				self.DQN_Model = copy.deepcopy(self.DQN_Best)
				self.learning_rate /= 2
				adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
				self.DQN_Model.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

				#EVALUATE THE MODEL AGAIN TO GET A TRUE REWARD SCORE VERSUS KEEPING IT THE SAME ALWAYS
				#DEALS WITH STOCHASTICITY IN ENVIRONMENT
				#Variables to store the outputs of policy progression through the environment
				state = None
				action = None
				reward = None
				next_state = None
				done = None
				not_termination = None
				#Numpy array to store rewards over testing per episode
				all_rewards = np.zeros(self.test_episodes, dtype=int)
				#Holds the TD Error sum and a counter for total steps
				steps = 0
				td_error_sum = 0
				for i in range(self.test_episodes):
					state = self.env.reset()
					done = False
					while(done == False):
						#Get the q-values using our policy and get an action
						q_values = self.DQN_Model.model.predict(np.array([state]), batch_size=1)
						action = self.greedy_policy(q_values)
						#Take our action to move to a next state, get a reward, and determine if our environment has terminated
						next_state, reward, done, _ = self.env.step(action[0])
						#Add the current reward to the data structure storing the rewards
						all_rewards[i] += reward
						#Add to the td_error_sum variable as well as the steps we have computed it over
						td_error_sum += abs((reward + (self.gamma * np.max(self.DQN_Model.model.predict(np.array([next_state]), batch_size=1)))) - np.max(q_values))
						steps += 1
						#Set the current state to the next state from taking the action for the next loop
						state = next_state
				#Compute the cumulative reward
				average_reward = np.mean(all_rewards)
				self.best_reward = average_reward
				self.best_td_error = td_error_sum/steps

			print("The average reward is: ", self.best_reward)
			print("The average TD error is: ", self.best_td_error)
			print()
			return self.best_reward, self.best_td_error
		else:
			print("The average reward is: ", average_reward)
			print("The average TD error is: ", td_error_sum/steps)
			print()
			return average_reward, td_error_sum/steps



	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		#Variables to store the outputs of policy progression through the environment
		state = None
		action = None
		reward = None
		next_state = None
		done = None
		not_termination = None

		#Progress through the environment using the initial policy for burn_in_value number of times
		for i in range(self.burn_in_value):
			#For the inital case or when a terminal state was encountered we need to reset the environment
			if((i == 0) or (done == True)):
				state = self.env.reset()
				done = False

			#Get the q-values using our policy and get an action
			q_values = self.DQN_Model.model.predict(np.array([state]), batch_size=1)
			action = self.epsilon_greedy_policy(q_values)
			#Take our action to move to a next state, get a reward, and determine if our environment has terminated
			next_state, reward, done, _ = self.env.step(action[0])
			#Set the termination variable as it will be helpful for calculating y_j
			if(done == False):
				not_termination = 1.0
			else:
				not_termination = 0.0
			#Create the tuple to store to the replay memory and append it
			tuple = [state, q_values, np.array(reward), next_state, np.array(not_termination)]
			self.Memory.append(tuple)

			#Set the current state to the next state from taking the action for the next loop
			state = next_state
		return



def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str, help='Either CartPole-v0 or MountainCar-v0')
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory.
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session.
	keras.backend.tensorflow_backend.set_session(sess)

	Agent = DQN_Agent(args.env)
	Agent.train()


if __name__ == '__main__':
	main(sys.argv)
