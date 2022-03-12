import os
import pickle
import random
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import settings

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


#factors determining explorative behaviour
epsilon = 1.0
epsilon_decay = 0.995
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
fc1Dim = featureSum
fc2Dim = 256


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


def build_model(lr = learning_rate, inputDim=featureSum, fc1Dim= fc1Dim, fc2Dim=fc2Dim,
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
    if self.train and not os.path.isfile(
            "agent_code//dnq_agent//model.pt"):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        self.model = build_model()

    elif self.train and os.path.isfile("agent_code//dnq_agent//model.pt"):
        print("Loading model from saved state.")
        self.model = T.load("model.pt")


    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        self.model = T.load(
            "model.pt")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() <= epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return ACTIONS[T.argmax(self.model.forward(state_to_features(game_state)))]

def parseStep(game_state: dict) -> int:
    try:
        return game_state["step"]
    except ValueError:
        return 0


def parseField(game_state: dict) -> np.array:
    try:
        return game_state["field"].reshape(-1)
    except ValueError:
        return np.zeros(fieldNum)


def parseExplosionMap(game_state: dict) -> np.array:
    try:
        return game_state["explosion_map"].reshape(-1)
    except ValueError:
        return np.zeros(explosion_mapNum)


def parseCombinedFieldExplosionMap(game_state: dict) -> np.array:
    try:
        return parseField(game_state) - 2 * parseExplosionMap(game_state)
    except ValueError:
        return np.zeros(explosion_mapNum)


def parseBombs(game_state: dict) -> np.array:
    try:
        bombVector = []
        for bomb in game_state["bombs"]:
            bombVector.append(bomb[0][0])
            bombVector.append(bomb[0][1])
            bombVector.append(bomb[1])

        while len(bombVector) < bombsNum:
            bombVector.append(0)

        return np.array(bombVector)
    except ValueError:
        return np.zeros(bombsNum)


def parseCoins(game_state: dict) -> np.array:
    try:
        coinVector = []
        for coin in game_state["coins"]:
            coinVector.append(coin[0])
            coinVector.append(coin[1])

        while len(coinVector) < coinsNum:
            coinVector.append(0)

        return np.array(coinVector)
    except ValueError:
        return np.zeros(coinsNum)


def parseSelf(game_state: dict) -> np.array:
    try:
        selfVector = np.array([game_state["self"][2],
                               game_state["self"][3][0],
                               game_state["self"][3][1]])
        return selfVector
    except ValueError:
        return np.zeros(3)


def parseOthers(game_state: dict) -> np.array:
    try:
        otherVector = []
        for other in game_state["others"]:
            otherVector.append(other[2])
            otherVector.append(other[3][0])
            otherVector.append(other[3][1])

        while len(otherVector) < othersNum:
            otherVector.append(0)

        return np.array(otherVector)

    except ValueError:
        return np.zeros(othersNum)


def state_to_features(game_state: dict) -> np.array:
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
        return None

    featurevector = np.zeros(featureSum)
    temp = 0
    featurevector[temp:temp+fieldNum] = parseCombinedFieldExplosionMap(game_state)
    temp += fieldNum
    featurevector[temp:temp+bombsNum] = parseBombs(game_state)
    temp += bombsNum
    featurevector[temp:temp+coinsNum] = parseCoins(game_state)
    temp += coinsNum
    featurevector[temp:temp+3] = parseSelf(game_state)
    temp += 3
    featurevector[temp:temp+othersNum] = parseOthers(game_state)

    return T.tensor(featurevector).float()
