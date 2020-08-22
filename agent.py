#Developer: Dillon Pulliam
#Date: 8/21/2020
#Purpose: The purpose of this file is specify the DQN Agent


#Imports needed
import os
import time
import copy
import gym
import numpy as np
import matplotlib.pyplot as plt


#Local imports
from replay_memory import *
from q_network import *


#DQN Agent class definition
class DQN_Agent():
	#Name:          __init__
	#Purpose:       define the DQN agent
	#Inputs:        environment_name -> name of the environment the model will be used in
	#               render -> whether to render the environment
	#Output:        none -> just defines the DQN agent
	def __init__(self, environment_name, render=False):
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
		return

	#Name:          epsilon_greedy_policy
	#Purpose:       sample random actions in an epsilon greedy fashion for training (exploration vs exploitation)
	#Inputs:        q_values -> q-values output from the deep q-network
	#Output:        actions -> actions based on epsilon greedy sampling
	def epsilon_greedy_policy(self, q_values):
		#Determine the number of actions to get and compute random actions and greedy actions
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

	#Name:          greedy_policy
	#Purpose:       sample random actions in a greedy fashion for testing
	#Inputs:        q_values -> q-values output from the deep q-network
	#Output:        actions -> actions based on greedy sampling
	def greedy_policy(self, q_values):
		actions = np.argmax(q_values, axis=1)
		return actions

	#Name:          train
	#Purpose:       train the DQN agent in the environment
	#Inputs:        none
	#Output:        none
	def train(self):
		#Execute burn-in to fill the memory buffer
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

		#Loop through all episodes training the agent
		for i in range(self.episodes):
			#Reset the environment state, set done to false, and while not done take actions in the environment
			state = self.env.reset()
			done = False
			while(done == False):
				#Get the q-values using our policy, get an action using epsilon greedy, and take the action
				#to move to a next state, get a reward, and determine if the environment has terminated
				q_values = self.DQN_Model.model.predict(np.array([state]), batch_size=1)
				action = self.epsilon_greedy_policy(q_values)
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
				#Store the specific episode we are testing on, the test reward itself, and the TD error
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

	#Name:          test_video
	#Purpose:       save videos of the agent acting in the environment
	#Inputs:        epi -> episode number on
	#Output:        none -> just saves a video of the agent acting in the environment
	def test_video(self, epi):
		#Create a new environment and a save path
		new_environment = gym.make(self.environment_name)
		save_path = "./videos-%s-%s" % (self.environment_name, epi)
		if not os.path.exists(save_path):
			os.mkdir(save_path)

		#Create a video of the agent acting in the environment
		env = gym.wrappers.Monitor(new_environment, save_path, force=True)
		reward_total = []
		state = new_environment.reset()

		#Act in the environment until a terminal state is reached
		done = False
		while not done:
			new_environment.render()
			q_values = self.DQN_Model.model.predict(np.array([state]), batch_size=1)
			action = self.greedy_policy(q_values)
			next_state, reward, done, info = new_environment.step(action[0])
			state = next_state
			reward_total.append(reward)
		env.close()
		new_environment.close()
		return

	#Name:          plot_graphics
	#Purpose:       plot the results of training the DQN agent in the environment
	#Inputs:        average_rewards -> test rewards of the agent throughout training
	#				average_td_error -> td-error of the agent throughout training
	#Output:        none -> just plots the training results
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

	#Name:          test
	#Purpose:       test the DQN agent in the environment
	#Inputs:        none
	#Outputs:        average_reward -> average cumulative test reward of the DQN agent
	#				average_td_error -> average td-error of the DQN agent
	def test(self):
		#Variables to store the outputs of policy progression through the environment
		state = None
		action = None
		reward = None
		next_state = None
		done = None
		not_termination = None

		#Numpy array to store rewards over testing per episode, variables to hold the td-error sum and total step counter
		all_rewards = np.zeros(self.test_episodes, dtype=int)
		steps = 0
		td_error_sum = 0

		#Loop through all test episodes evaluating performance
		for i in range(self.test_episodes):
			#Reset the state, set done to false, and act in the environment
			state = self.env.reset()
			done = False
			while(done == False):
				#Get the q-values using our policy, get an action using greedy, and take the action moving to a next state, getting a reward,
				#and determining if the environment has terminated
				q_values = self.DQN_Model.model.predict(np.array([state]), batch_size=1)
				action = self.greedy_policy(q_values)
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
			#Only save the model if it performs better; if the model fails to perform better re-load the best model, half the learning rate,
			#and re-compile the model
			if(average_reward >= self.best_reward):
				self.DQN_Best = copy.deepcopy(self.DQN_Model)
				self.best_reward = average_reward
				self.best_td_error = td_error_sum/steps
			else:
				self.DQN_Model = copy.deepcopy(self.DQN_Best)
				self.learning_rate /= 2
				adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
				self.DQN_Model.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

				#Evaluate the model again to get a true reward score versus keeping it the same; deals with environment stochasticity
				#Reset the all_rewards, steps, and td_error_sum variables
				all_rewards = np.zeros(self.test_episodes, dtype=int)
				steps = 0
				td_error_sum = 0

				#Loop through all test episodes evaluating performance
				for i in range(self.test_episodes):
					#Reset the state, set done to false, and act in the environment
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


	#Name:          burn_in_memory
	#Purpose:       burn in the replay memory by iterating through a certain number of episodes / transitions
	#Inputs:        none
	#Output:        none -> just performs burn in
	def burn_in_memory(self):
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
