#Developer: Dillon Pulliam
#Date: 8/22/2020
#Purpose: Main file to train the DQN Agent


#Imports needed
import sys 
import keras
import tensorflow as tf


#Local imports
from agent import *
from utils import *


#Main function
def main(args):
	#Parse the command line arguements and get the environment name
	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory.
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session.
	keras.backend.tensorflow_backend.set_session(sess)

	#Initialize and train the agent
	Agent = DQN_Agent(args.env)
	Agent.train()
	return


if __name__ == '__main__':
	main(sys.argv)
