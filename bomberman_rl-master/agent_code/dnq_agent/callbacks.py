import os
import pickle
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from collections import deque
import settings
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.grid import Grid



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


#factors determining explorative behaviour
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.05


#model parameters
action_size = len(ACTIONS)
gamma = 0.99
learning_rate = 0.0001
input_dims = 5
fc_input_dims = 720
fc1_dims = 360
fc2_dims = 180


class Model(nn.Module):
    def __init__(self, lr, input_dims, fc_input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.fc_input_dims = fc_input_dims
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(self.input_dims, 4 * self.input_dims, 5)
        self.pool1 = nn.MaxPool2d((2, 2), ceil_mode=True)
        self.conv2 = nn.Conv2d(4 * self.input_dims, 16 * self.input_dims, 3)
        self.pool2 = nn.MaxPool2d((2, 2), ceil_mode=True)

        self.fc1 = nn.Linear(self.fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if False else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.float()
        x = F.relu(self.conv1(state))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.reshape(-1, 720)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


def build_model(lr = learning_rate, inputDim= input_dims, fc_input_dims = fc_input_dims, fc1Dim= fc1_dims, fc2Dim=fc2_dims,
                n_actions=len(ACTIONS)):
    return Model(lr = lr, input_dims = inputDim, fc_input_dims = fc_input_dims,
                 fc1_dims= fc1Dim, fc2_dims = fc2Dim, n_actions = n_actions), \
           Model(lr = lr, input_dims = inputDim, fc_input_dims = fc_input_dims,
                 fc1_dims= fc1Dim, fc2_dims = fc2Dim, n_actions = n_actions)


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    evalmodelname = "evalmodel.pt"
    evalmodelpath = os.path.join(os.getcwd(), evalmodelname)

    nextmodelname = "nextmodel.pt"
    nextmodelpath = os.path.join(os.getcwd(), nextmodelname)


    if self.train and not os.path.isfile(os.path.join(os.getcwd(), evalmodelname)):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        self.nextmodel, self.evalmodel = build_model()

    elif self.train and os.path.isfile(os.path.join(os.getcwd(), evalmodelname)):
        print("Loading model from saved state.")
        self.evalmodel = T.load(evalmodelpath)
        self.nextmodel = T.load(nextmodelpath)

    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        self.evalmodel = T.load(evalmodelpath)
        self.nextmodel = T.load(nextmodelpath)


