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
epsilon_decay = 0.99995
epsilon_min = 0.05


#model parameters
action_size = len(ACTIONS)
gamma = 0.80
learning_rate = 0.0001
fc1Dim = 9
fc2Dim = 8
input_dims = 10

class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc_input_dims = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if False else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        actions = self.fc2(x)

        return actions


def build_model(lr = learning_rate, inputDim= input_dims, fc1Dim= fc1Dim, fc2Dim=fc2Dim,
                n_actions=len(ACTIONS)):
    return Model(lr = lr, input_dims = inputDim, fc1_dims = fc1Dim, fc2_dims = fc2Dim, n_actions = n_actions), \
           Model(lr = lr, input_dims = inputDim, fc1_dims = fc1Dim, fc2_dims = fc2Dim, n_actions = n_actions)


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

    self.finder = AStarFinder()


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
        action_chosen = np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
        #print("Action:", action_chosen)
        #action_chosen = rolemodel.act(self, game_state)

        return action_chosen
    #self.logger.debug("Querying model for action.")

    """
    inputvec = state_to_features(game_state)    
    print("Input vector :", inputvec)
    print("prediction of q values: ", self.model.forward(inputvec))
    """

    #predicted_q_values = self.evalmodel.forward(state_to_features(game_state))
    #action_chosen = ACTIONS[T.argmax(predicted_q_values)]
    #print("Action: ", action_chosen)

    return directionNextCoin(self, game_state)


def directionNextCoin(self, game_state: dict):
    s0, s1 = game_state["self"][3]
    map = game_state["field"] + game_state["explosion_map"]
    closest_coin = sorted(game_state["coins"], key=lambda x: (s0 - x[0]) ** 2 + (s1 - x[1]) ** 2)[0]
    map = abs(map)*(-1) + 1
    grid = Grid(matrix=map)
    start = grid.node(s0, s1)
    end = grid.node(closest_coin[0], closest_coin[1])
    path, runs = self.finder.find_path(start, end, grid)
    next_tile = path[1]  #first element is starting point therefore the second element is the successor
    if next_tile[0] - s0 == 1:
        return "RIGHT"
    if next_tile[0] - s0 == -1:
        return "LEFT"
    if next_tile[1] - s1 == 1:
        return "DOWN"
    if next_tile[1] - s0 == -1:
        return "UP"


def validAction(game_state: dict):
    validAction = [1, 1, 1, 1, 1, 1]

    #retrieve player position
    playerx, playery = game_state["self"][3]

    #UP -- Check for wall or crate or explosion
    if (game_state["field"][playerx][playery-1] != 0) or isdeadly(playerx, playery-1, game_state):
        validAction[0] = 0
    #RIGHT -- Check for wall or crate or explosion
    if game_state["field"][playerx+1][playery] != 0 or isdeadly(playerx+1, playery, game_state):
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
        returnarray = T.zeros((11, 11))
    else:
        returnarray = -T.ones((11, 11))

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


def parseField(game_state: dict, feature_vector):
    try:
        playerx, playery = game_state["self"][3]
        ##Above agent
        feature_vector[1] = game_state["field"][playerx][playery-1]
        ##Right to agent
        feature_vector[2] = game_state["field"][playerx+1][playery]
        ##Below Agent
        feature_vector[3] = game_state["field"][playerx][playery+1]
        ##Left to Agent
        feature_vector[4] = game_state["field"][playerx-1][playery]
        ##Upperleft of agent
        feature_vector[5] = game_state["field"][playerx-1][playery - 1]
        ##Upperright to agent
        feature_vector[6] = game_state["field"][playerx + 1][playery - 1]
        ##Lowerleft to Agent
        feature_vector[7] = game_state["field"][playerx - 1][playery + 1]
        ##Lowerright to Agent
        feature_vector[8] = game_state["field"][playerx + 1][playery + 1]

    except ValueError:
        print("Value error in field parser")


def parseExplosionMap(game_state: dict, feature_vector):
    try:
        playerx, playery = game_state["self"][3]
        ##Agents position
        feature_vector[0] = 3 if game_state["explosion_map"][playerx][playery - 1] != 0 else feature_vector[0]
        ##Above agent
        feature_vector[1] = 3 if game_state["explosion_map"][playerx][playery - 1] != 0 else feature_vector[1]
        ##Right to agent
        feature_vector[2] = 3 if game_state["explosion_map"][playerx + 1][playery] != 0 else feature_vector[2]
        ##Below Agent
        feature_vector[3] = 3 if game_state["explosion_map"][playerx][playery + 1] != 0 else feature_vector[3]
        ##Left to Agent
        feature_vector[4] = 3 if game_state["explosion_map"][playerx - 1][playery] != 0 else feature_vector[4]
        ##Upperleft of agent
        feature_vector[5] = 3 if game_state["explosion_map"][playerx - 1][playery - 1] != 0 else feature_vector[5]
        ##Upperright to agent
        feature_vector[6] = 3 if game_state["explosion_map"][playerx + 1][playery - 1] != 0 else feature_vector[6]
        ##Lowerleft of Agent
        feature_vector[7] = 3 if game_state["explosion_map"][playerx - 1][playery + 1] != 0 else feature_vector[7]
        ##Lowerright to Agent
        feature_vector[8] = 3 if game_state["explosion_map"][playerx + 1][playery + 1] != 0 else feature_vector[8]
    except ValueError:
        print("Value error in explosion map parser")


def parseBombs(game_state: dict, feature_vector):
    try:
        deadly_map = np.zeros((17, 17))
        for bomb in game_state["bombs"]:
            bombx, bomby = bomb[0]
            #On bombsite
            deadly_map[bombx, bomby] = bomb[1] + 1
            ##Tile is above bomb
            if game_state["field"][bombx][bomby - 1] != -1:
                deadly_map[bombx, bomby-3:bomby] = bomb[1] + 1
            ##Tile is below bomb
            if game_state["field"][bombx][bomby + 1] != -1:
                deadly_map[bombx, bomby:bomby+3] = bomb[1] + 1
            ##Tile is left to bomb
            if game_state["field"][bombx - 1][bomby] != -1:
                deadly_map[bombx-3:bombx, bomby] = bomb[1] + 1
            ##Tile is right to bomb
            if game_state["field"][bombx + 1][bomby] != -1:
                deadly_map[bombx:bombx+3, bomby] = bomb[1] + 1
        playerx, playery = game_state["self"][3]
        ##On agent position
        feature_vector[0] = 4 if deadly_map[playerx][playery] != 0 else feature_vector[0]
        ##Above agent
        feature_vector[1] = 4 if deadly_map[playerx][playery - 1] != 0 else feature_vector[1]
        ##Right to agent
        feature_vector[2] = 4 if deadly_map[playerx + 1][playery] != 0 else feature_vector[2]
        ##Below Agent
        feature_vector[3] = 4 if deadly_map[playerx][playery + 1] != 0 else feature_vector[3]
        ##Left to Agent
        feature_vector[4] = 4 if deadly_map[playerx - 1][playery] != 0 else feature_vector[4]
        ##Upperleft to agent
        feature_vector[5] = 4 if deadly_map[playerx-1][playery - 1] != 0 else feature_vector[5]
        ##Upperright to agent
        feature_vector[6] = 4 if deadly_map[playerx + 1][playery-1] != 0 else feature_vector[6]
        ##Lowerleft to Agent
        feature_vector[7] = 4 if deadly_map[playerx-1][playery + 1] != 0 else feature_vector[7]
        ##Lowerright to Agent
        feature_vector[8] = 4 if deadly_map[playerx + 1][playery + 1] != 0 else feature_vector[8]
    except ValueError:
        print("Value Error in bomb parser")

def parseBombPossible(game_state: dict, feature_vector):
    if game_state["self"][2]:
        feature_vector[9] = 1
    else:
        feature_vector[9] = 0


def parseCoins(game_state: dict, feature_vector):
    try:
        coinmap = np.zeros((17, 17))
        for coin in game_state["coins"]:
            coinmap[coin[0]][coin[1]] = 1
        playerx, playery = game_state["self"][3]
        ##Above agent
        feature_vector[1] = 2 if coinmap[playerx][playery - 1] == 1 else feature_vector[1]
        ##Right to agent
        feature_vector[2] = 2 if coinmap[playerx + 1][playery] == 1 else feature_vector[2]
        ##Below Agent
        feature_vector[3] = 2 if coinmap[playerx][playery + 1] == 1 else feature_vector[3]
        ##Left to Agent
        feature_vector[4] = 2 if coinmap[playerx - 1][playery] == 1 else feature_vector[4]
        ##Upperleft to agent
        feature_vector[5] = 2 if coinmap[playerx - 1][playery - 1] == 1 else feature_vector[5]
        ##Upperright to agent
        feature_vector[6] = 2 if coinmap[playerx + 1][playery - 1] == 1 else feature_vector[6]
        ##Lowerleft Agent
        feature_vector[7] = 2 if coinmap[playerx - 1][playery + 1] == 1 else feature_vector[7]
        ##Lowerright to Agent
        feature_vector[8] = 2 if coinmap[playerx + 1][playery + 1] == 1 else feature_vector[8]
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


def parseOthers(game_state: dict, feature_vector):
    try:
        othersmap = np.zeros((17, 17))
        for other in game_state["others"]:
            othersmap[other[3][0]][other[3][1]] = 1
        playerx, playery = game_state["self"][3]
        ##Above agent
        feature_vector[1] = 6 if othersmap[playerx][playery - 1] == 1 else feature_vector[1]
        ##Right to agent
        feature_vector[2] = 6 if othersmap[playerx + 1][playery] == 1 else feature_vector[2]
        ##Below Agent
        feature_vector[3] = 6 if othersmap[playerx][playery + 1] == 1 else feature_vector[3]
        ##Left to Agent
        feature_vector[4] = 6 if othersmap[playerx - 1][playery] == 1 else feature_vector[4]
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
        return T.zeros(10)
    feature_vector = T.zeros(10) #represent the tiles agentpos, up, right, down, left relative to the agent
    ##upperleft, upperright, lowerleft, lowerright
    parseField(game_state, feature_vector)
    parseCoins(game_state, feature_vector)
    parseOthers(game_state, feature_vector)
    parseBombs(game_state, feature_vector)
    parseExplosionMap(game_state, feature_vector)
    parseBombPossible(game_state, feature_vector)

    return feature_vector
