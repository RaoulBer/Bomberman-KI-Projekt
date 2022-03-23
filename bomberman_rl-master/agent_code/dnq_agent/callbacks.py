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
epsilon_decay = 0.9995
epsilon_min = 0.05


#model parameters
action_size = len(ACTIONS)
gamma = 0.99
learning_rate = 0.0001
fc1Dim = 256
fc2Dim = 128
input_dims = 576

class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc_input_dims = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.conv1 = nn.Conv2d(2,32,kernel_size=3)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3)

        self.pool = nn.MaxPool2d((2,2), ceil_mode=True)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.conv1(state)
        x = self.pool(x)
        x = self.conv2(x)
        #x = self.pool(x)
        x = x.reshape(-1, 576)
        x = F.relu(self.fc1(x))
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
        #ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
        #valids = validAction(game_state)
        #possible_actions = [a for idx, a in enumerate(ACTIONS) if valids[idx]]
        #action_chosen = np.random.choice(possible_actions)
        action_chosen = np.random.choice(ACTIONS, p=[0.2,0.2,0.2,0.2,0.1,0.1])
        #print("Action:", action_chosen)

        return action_chosen
    #self.logger.debug("Querying model for action.")

    """
    inputvec = state_to_features(game_state)    
    print("Input vector :", inputvec)
    print("prediction of q values: ", self.model.forward(inputvec))
    """

    predicted_q_values = self.model.forward(state_to_features(game_state))
    action_chosen = ACTIONS[T.argmax(predicted_q_values)]
    #print("Action: ", action_chosen)

    return action_chosen

def validAction(game_state: dict):
    validAction = [1, 1, 1, 1, 1, 1]

    #retrieve player position
    playerx, playery = game_state["self"][3]

    #UP -- Check for wall or crate or explosion
    if (game_state["field"][playerx][playery-1] != 0) or isdeadly(playerx,playery-1, game_state):
        validAction[0] = 0
    #RIGHT -- Check for wall or crate or explosion
    if game_state["field"][playerx+1][playery] != 0 or isdeadly(playerx+1,playery, game_state):
        validAction[1] = 0
    #DOWN -- Check for wall or crate or explosion
    if game_state["field"][playerx][playery+1] != 0 or isdeadly(playerx, playery+1, game_state):
        validAction[2] = 0
    #LEFT -- Check for wall or crate or explosion
    if game_state["field"][playerx-1][playery] != 0 or isdeadly(playerx-1, playery, game_state):
        validAction[3] = 0

    #Check if Bomb action is possible
    if not game_state["self"][2]:
        validAction[5] = 0

    return np.array(validAction)

def isdeadly(x,y,game_state: dict):
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

def aroundAgent(game_state: dict, input_field, isbombarray = False) -> T.tensor:
    #returns a 5 +/- array of the input_field around the agents position
    playerx, playery = game_state["self"][3]
    if isbombarray:
        returnarray = T.zeros((11,11))
    else:
        returnarray = -T.ones((11,11))

    for i in range(-5,6):
        for j in range(-5,6):
            tempx = playerx + i
            tempy = playery + j
            if (tempx > 0 and tempx < 17) and (tempy > 0 and tempy < 17):
                returnarray[i+5][j+5] = input_field[tempx][tempy]

    return returnarray

def parseStep(game_state: dict) -> int:
    try:
        return game_state["step"]
    except ValueError:
        return 0


def parseField(game_state: dict) -> T.tensor:
    try:
        return T.tensor(game_state["field"], dtype=T.float)
    except ValueError:
        print("Value error in field parser")


def parseExplosionMap(game_state: dict) -> T.tensor:
    try:
        return T.tensor(game_state["explosion_map"])
    except ValueError:
        print("Value error in explosion map parser")


def parseBombs(game_state: dict, deadly_map):
    try:
        for bomb in game_state["bombs"]:
            bombx, bomby = bomb[0]
            #On bombsite
            deadly_map[bombx,bomby] = -(bomb[1] - 1) / 4
            ##Tile is above bomb
            if game_state["field"][bombx][bomby-1] != -1:
                deadly_map[bombx,bomby-3:bomby] = -(bomb[1] - 1)/4
            ##Tile is below bomb
            if game_state["field"][bombx][bomby+1] != -1:
                deadly_map[bombx,bomby:bomby+3] = -(bomb[1] - 1)/4
            ##Tile is left to bomb
            if game_state["field"][bombx-1][bomby] != -1:
                deadly_map[bombx-3:bombx,bomby] = -(bomb[1] - 1)/4
            ##Tile is right to bomb
            if game_state["field"][bombx+1][bomby] != -1:
                deadly_map[bombx:bombx+3,bomby] = -(bomb[1] - 1)/4

    except ValueError:
        print("Value Error in bomb parser")


def parseCoins(game_state: dict, objective_map):
    try:
        for coin in game_state["coins"]:
            objective_map[coin[0]][coin[1]] = 0.1
    except ValueError:
        print("Value Error in coin parser")

def parseSelf(game_state: dict, objective_map):
    try:
        if game_state["self"][2]:
            objective_map[game_state["self"][3][0]][game_state["self"][3][1]] = 0.7
        else:
            objective_map[game_state["self"][3][0]][game_state["self"][3][1]] = -0.7
    except ValueError:
        print("Error in Self parser")


def parseOthers(game_state: dict, objective_map):
    try:
        for other in game_state["others"]:
            objective_map[other[3][0]][other[3][1]] = 0.5

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

    objective_map = parseField(game_state)
    parseCoins(game_state, objective_map)
    parseSelf(game_state, objective_map)
    parseOthers(game_state, objective_map)

    deadly_map = parseExplosionMap(game_state)
    parseBombs(game_state, deadly_map)
    #validactions = validAction(game_state)

    objective_map = aroundAgent(game_state, objective_map, False)
    deadly_map = aroundAgent(game_state, deadly_map, True)

    #output_vector = np.append(objective_map.flatten(), deadly_map.flatten())
    #output_vector = np.append(output_vector, validactions)
    returnvariable = T.stack((objective_map, deadly_map)).float()
    return returnvariable
