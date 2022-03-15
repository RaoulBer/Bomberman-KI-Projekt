import os
import pickle
import random
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools

import settings

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


#factors determining explorative behaviour
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.10


#feature counting
roundNum = 1
stepNum = 1
fieldNum = settings.ROWS * settings.COLS
bombsNum = 3 * 4  # ((x,y)t) * number of max. active bombs
explosion_mapNum = fieldNum # same number of parameters
coinsNum = 18 # 9 * 2 coordinates
selfNum = 3 # bomb is possible plus x and y coordinate
othersNum = 3 * 3 # bomb is possible plus x and y coordinate times 3
featureSum = fieldNum + bombsNum + coinsNum + selfNum + othersNum


#model parameters
action_size = len(ACTIONS)
gamma = 0.90
learning_rate = 0.0001
fc1Dim = int((settings.ROWS * settings.COLS) / 2)
fc2Dim = int((settings.ROWS * settings.COLS) / 4)
input_dims = (settings.ROWS * settings.COLS)


class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(Model, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


def build_model(lr = learning_rate, inputDim= input_dims, fc1Dim= fc1Dim, fc2Dim=fc2Dim,
                n_actions=len(ACTIONS)):
    return Model(lr = lr, input_dims = inputDim, fc1_dims = fc1Dim, fc2_dims = fc2Dim, n_actions = n_actions)


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
        return np.random.choice([a for idx, a in enumerate(ACTIONS) if validAction(game_state)[idx]])

    #self.logger.debug("Querying model for action.")

    """
    inputvec = state_to_features(game_state)    
    print("Input vector :", inputvec)
    print("prediction of q values: ", self.model.forward(inputvec))
    """

    predicted_q_values = self.model.forward(state_to_features(game_state))
    action_chosen = ACTIONS[T.argmax(excludeInvalidActions(game_state,predicted_q_values))]
    print("Action: ", action_chosen)

    return action_chosen

def validAction(game_state: dict):
    validAction = [True, True, True, True, True, True]
    playerx, playery = game_state["self"][3]

    #UP
    if (game_state["field"][playerx][playery-1] != 0):
        validAction[0] = False
    #RIGHT
    if (game_state["field"][playerx+1][playery] != 0):
        validAction[1] = False
    #DOWN
    if (game_state["field"][playerx][playery+1] != 0):
        validAction[2] = False
    #LEFT
    if (game_state["field"][playerx-1][playery] != 0):
        validAction[3] = False

    validAction[5] = game_state["self"][2]

    return validAction

def excludeInvalidActions(game_state: dict, q_values_tensor):
    possibleList = validAction(game_state)
    for i in range(0, len(q_values_tensor)):
        if not possibleList[i]:
            q_values_tensor[i] = float(-np.inf)

    return q_values_tensor

def parseStep(game_state: dict) -> int:
    try:
        return game_state["step"]
    except ValueError:
        return 0


def parseField(game_state: dict) -> np.array:
    try:
        return game_state["field"]
    except ValueError:
        print("Value error in field parser")


def parseExplosionMap(game_state: dict) -> np.array:
    try:
        return game_state["explosion_map"]
    except ValueError:
        print("Value error in explosion map parser")


def parseCombinedFieldExplosionMap(game_state: dict) -> np.array:
    try:
        return parseField(game_state) - 2 * parseExplosionMap(game_state)
    except ValueError:
        print("Value error in combined field and explosion map parser")


def parseBombs(game_state: dict, featurevector) -> np.array:
    try:
        for bomb in game_state["bombs"]:
            featurevector[bomb[0][0]][bomb[0][1]] = -10 * bomb[1]

    except ValueError:
        print("Value Error in bomb parser")


def parseCoins(game_state: dict, featurevector) -> np.array:
    try:
        for coin in game_state["coins"]:
            featurevector[coin[0]][coin[1]] = 10

    except ValueError:
        print("Value Error in coin parser")

def parseSelf(game_state: dict, featurevector) -> np.array:
    try:
        featurevector[game_state["self"][3][0]][game_state["self"][3][1]] = 100 if game_state["self"][2] else -100

    except ValueError:
        print("Error in Self parser")


def parseOthers(game_state: dict, featurevector) -> np.array:
    try:
        for other in game_state["others"]:
            featurevector[other[3][0]][other[3][1]] = 60 if game_state["self"][2] else -60

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

    featurevector = parseCombinedFieldExplosionMap(game_state)
    parseBombs(game_state, featurevector)
    parseCoins(game_state, featurevector)
    parseSelf(game_state, featurevector)
    parseOthers(game_state, featurevector)

    return T.tensor(featurevector.flatten()).float()
