# Deep_Q-Network

## Virtual environment creation:
Assuming [Anaconda](https://www.anaconda.com/) is installed the following steps can be taken to create a conda virtual environment for this project. Note that the local system used for development had access to a GeForce GTX 1060 GPU with CUDA version 10.2, thus the PyTorch install command may vary based on CUDA version. Please see [PyTorch installation](https://pytorch.org/) for more details.
```
conda create -n dqn python=3.7
conda activate car_classification
conda install -c conda-forge keras
pip install gym
conda install matplotlib 
```
