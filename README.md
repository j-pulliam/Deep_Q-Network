# Deep_Q-Network
In this project we build and train a [Deep Q-Network](https://arxiv.org/abs/1312.5602) for both the [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) and [MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/) environments in [OpenAI Gym](https://gym.openai.com/). These type of networks were introduced in the ground-breaking paper ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602) published by [DeepMind](https://deepmind.com/) in 2013. The key idea is to use a neural network as the replacement for the Q-Table alleviating the scaling problem that occurs in environments with larger state / action spaces.

<img src="https://github.com/j-pulliam/Deep_Q-Network/blob/master/img/DQN.jpg">
[Source](https://rubikscode.net/2019/07/08/deep-q-learning-with-python-and-tensorflow-2-0/)

Using the default hyper-parameters and running the code "as-is" should lead directly to convergence for both environments. Feel free to modify the code and experiment with other extensions to the DQN model such as [Double DQN](https://arxiv.org/abs/1509.06461), [Dueling DQN](https://arxiv.org/abs/1511.06581), [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), etc.    

## Virtual environment creation:
Assuming [Anaconda](https://www.anaconda.com/) is installed the following steps can be taken to create a conda virtual environment for this project. Note that the local system used for development had access to a GeForce GTX 1060 GPU with CUDA version 10.2.
```
conda create -n dqn python=3.7
conda activate dqn
conda install -c conda-forge keras
pip install gym
conda install matplotlib
```

## Training DQN for CartPole
To train a Deep Q-Network for the CartPole environment please execute the following command:
```
python main.py --env CartPole-v0
```

## CartPole Results


## Training DQN for MountainCar
To train a Deep Q-Network for the MountainCar environment please execute the following command:
```
python main.py --env MountainCar-v0
```

## MountainCar Results
