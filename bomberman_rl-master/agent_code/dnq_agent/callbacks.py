import os
import pickle
import random
import numpy as np

import torch as T
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools

import settings

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


#factors determining explorative behaviour
epsilon = 1.0
epsilon_decay = 0.99995
epsilon_min = 0.10


#model parameters
action_size = len(ACTIONS)
gamma = 0.90
learning_rate = 0.0001
fc1Dim = 512
fc2Dim = 256
kernel_size = 5
input_channels = 4 #how many feature maps in nn (static, dynamic, deadly)=3

class Model(nn.Module):
    def __init__(self, lr, input_channels, kernel_size, fc1_dims, fc2_dims,
                 n_actions):
        super(Model, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0, "Kernel size has to be an odd integer"
        self.conv_output_dims = (17 - 2*(self.kernel_size - 1))**2
        self.fc_input_dims = 1296
        self.n_actions = n_actions
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(self.input_channels, 2 * self.input_channels, self.kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2 * input_channels, 4 * input_channels, self.kernel_size)
        self.fc1 = nn.Linear(self.fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 1296)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


def build_model(lr=learning_rate, input_channels=input_channels, kernel_size=kernel_size,
                fc1Dim= fc1Dim, fc2Dim=fc2Dim, n_actions=len(ACTIONS)):
    return Model(lr=lr, input_channels=input_channels, kernel_size=kernel_size,
                 fc1_dims=fc1Dim, fc2_dims=fc2Dim, n_actions=n_actions)


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
    modelname = "model.pt"
    modelpath = os.path.join(os.getcwd(), modelname)

    if self.train and not os.path.isfile(os.path.join(os.getcwd(), modelname)):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        self.model = build_model()

    elif self.train and os.path.isfile(os.path.join(os.getcwd(), modelname)):
        print("Loading model from saved state.")
        self.model = T.load(modelpath)

    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        self.model = T.load(modelpath)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation

    if self.train and random.random() <= epsilon:
        #self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
        valids = validAction(game_state)
        possible_actions = [a for idx, a in enumerate(ACTIONS) if valids[idx]]
        action_chosen = np.random.choice(possible_actions)
        print("Action:", action_chosen)
        return action_chosen

    #self.logger.debug("Querying model for action.")

    """
    inputvec = state_to_features(game_state)    
    print("Input vector :", inputvec)
    print("prediction of q values: ", self.model.forward(inputvec))
    """
    valids = validAction(game_state)
    predicted_q_values = self.model.forward(state_to_features(game_state))
    excluded_q_values = excludeInvalidActions(game_state, predicted_q_values)
    action_chosen = ACTIONS[T.argmax(excluded_q_values)]
    print("Action: ", action_chosen)

    return action_chosen

def validAction(game_state: dict):
    validAction = [True, True, True, True, True, True]

    #retrieve player position
    playerx, playery = game_state["self"][3]

    bombcoords = [bomb[0] for bomb in game_state["bombs"]]

    #UP -- Check for wall or crate
    if (game_state["field"][playerx][playery-1] != 0) or ((playerx, playery-1) in bombcoords) or isdeadly(playerx,playery-1, game_state):
        validAction[0] = False
    #RIGHT -- Check for wall or crate
    if game_state["field"][playerx+1][playery] != 0 or ((playerx+1, playery) in bombcoords) or isdeadly(playerx+1,playery, game_state):
        validAction[1] = False
    #DOWN -- Check for wall or crate
    if game_state["field"][playerx][playery+1] != 0 or ((playerx, playery+1) in bombcoords) or isdeadly(playerx, playery+1, game_state):
        validAction[2] = False
    #LEFT -- Check for wall or crate
    if game_state["field"][playerx-1][playery] != 0 or ((playerx-1, playery) in bombcoords) or isdeadly(playerx-1, playery, game_state):
        validAction[3] = False

    #Check if Bomb action is possible
    validAction[5] = game_state["self"][2]

    return validAction

def isdeadly(x,y, game_state: dict):
    if game_state["explosion_map"][x][y] != 0:
        return True
    else:
        return False

def excludeInvalidActions(game_state: dict, q_values_tensor):
    possibleList = validAction(game_state)
    excluded_qs = T.zeros_like(q_values_tensor)
    for idx, q in enumerate(q_values_tensor[0]):
        if not possibleList[idx]:
            excluded_qs[0][idx] = float(-np.inf)
        else:
            excluded_qs[0][idx] = q

    return excluded_qs

def parseStep(game_state: dict) -> int:
    try:
        return game_state["step"]
    except ValueError:
        return 0


def parseField(game_state: dict) -> T.tensor:
    try:
        return T.tensor(game_state["field"])
    except ValueError:
        print("Value error in field parser")


def parseExplosionMap(game_state: dict) -> T.tensor:
    try:
        return T.tensor(game_state["explosion_map"])
    except ValueError:
        print("Value error in explosion map parser")


def parseBombs(game_state: dict) -> T.tensor:
    try:
        bomb_tensor = T.zeros(settings.COLS, settings.ROWS)
        for bomb in game_state["bombs"]:
            bomb_tensor[bomb[0][0]][bomb[0][1]] = -1 * bomb[1] - 2
        return bomb_tensor
    except ValueError:
        print("Value Error in bomb parser")


def parseCoins(game_state: dict) -> T.tensor:
    try:
        coin_tensor = T.zeros(settings.ROWS, settings.COLS)
        for coin in game_state["coins"]:
            coin_tensor[coin[0]][coin[1]] = 1
        return coin_tensor
    except ValueError:
        print("Value Error in coin parser")

def parseSelf(game_state: dict) -> T.tensor:
    try:
        self_tensor = T.zeros(settings.ROWS, settings.COLS)
        self_tensor[game_state["self"][3][0]][game_state["self"][3][1]] = 1
        return self_tensor
    except ValueError:
        print("Error in Self parser")


def parseOthers(game_state: dict) -> T.tensor:
    try:
        others_tensor = T.zeros(settings.ROWS, settings.COLS)
        for other in game_state["others"]:
            others_tensor[other[3][0]][other[3][1]] = -1

        return others_tensor

    except ValueError:
        print("Value error in Others parser")

def state_to_features(game_state: dict):
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return T.tensor([0])

    #Subdividing into three object categories
    #bombs are on the negative number spectrum and the explosion of the positive part 0 indicating no threat
    deadly_tensor = parseExplosionMap(game_state) + parseBombs(game_state)

    #walls crates are -1, 1 free space is 0, coins have 2
    field_tensor = parseField(game_state)

    coin_tensor = parseCoins(game_state)

    #self (1) and others (2,3,4), empty is just 0
    dynamic_tensor = parseSelf(game_state) + parseOthers(game_state)

    feature_tensor = T.stack([deadly_tensor, field_tensor, coin_tensor, dynamic_tensor])

    return feature_tensor.float()
