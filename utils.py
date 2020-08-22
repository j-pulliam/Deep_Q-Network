#Developer: Dillon Pulliam
#Date: 8/22/2020
#Purpose: The purpose of this file is to specify utility functions needed by other files in this repo


#Imports needed
import argparse


#Name:          parse_arguments
#Purpose:       parse the command line arguements from the user
#Inputs:        none
#Output:        parser -> parser that contains the command line arguements from the user
def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str, help='Either CartPole-v0 or MountainCar-v0')
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()
