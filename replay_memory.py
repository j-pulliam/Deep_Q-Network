#Developer: Dillon Pulliam
#Date: 8/21/2020
#Purpose: The purpose of this file is specify the Replay Memory class


#Imports needed
import collections
import random
import numpy as np


#Replay Memory class definition
class Replay_Memory():
	#Name:          __init__
	#Purpose:       define the replay memory class which is used to store transitions recorded from the agent
	#Inputs:        memory_size -> max size of the replay memory after which old elements are replaced
	#               burn_in -> number of episodes that are written into the memory from the randomly initialized agent
	#Output:        none -> just defines the replay memory class
	def __init__(self, memory_size=50000, burn_in=10000):
		self.replay_memory = collections.deque(maxlen=memory_size)
		self.burn_in = burn_in
		return

	#Name:          sample_batch
	#Purpose:       returns a batch of randomly sampled transitions; tuple of (state, action, reward, next state, terminal flag)
	#Inputs:        batch_size -> batch size of random samples to return
	#Output:        batch -> batch of randomly sampled transitions
	def sample_batch(self, batch_size=32):
		r_batch = random.sample(self.replay_memory, batch_size)
		return np.array(r_batch)

	#Name:          append
	#Purpose:       appends transitions to the memory
	#Inputs:        transition -> new transition to append to the replay memory
	#Output:        none -> just appends transitions to the memory
	def append(self, transition):
		self.replay_memory.append(transition)
		return
