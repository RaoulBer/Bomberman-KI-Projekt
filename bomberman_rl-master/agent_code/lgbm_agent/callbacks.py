import os
import pickle
import random
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

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
learning_rate = 0.001
input_dims = (settings.ROWS * settings.COLS)
modelname = "model.pt"
modelpath = os.path.join(os.getcwd(), modelname)

class Model(MultiOutputRegressor):
    def __init__(self):
        super(MultiOutputRegressor).__init__(self, LGBMRegressor(n_estimators=100))
        self.isFit = False

def build_model():
    return MultiOutputRegressor(LGBMRegressor(n_estimators=100))

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

    if self.train and not os.path.isfile(os.path.join(os.getcwd(), modelname)):
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
        self.model = build_model()

    elif self.train and os.path.isfile(os.path.join(os.getcwd(), modelname)):
        print("Loading model from saved state.")
        with open(modelpath, "rb") as file:
            self.model = pickle.load(file)

    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        with open(modelpath, "rb") as file:
            self.model = pickle.load(file)


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
        randomaction = np.random.choice(ACTIONS)
        print("Random action", randomaction)
        return randomaction

    #self.logger.debug("Querying model for action.")

    """
    inputvec = state_to_features(game_state)    
    print("Input vector :", inputvec)
    print("prediction of q values: ", self.model.forward(inputvec))
    """
    try:
        predicted_q_values = self.model.predict(state_to_features(game_state))
    except:
        predicted_q_values = np.zeros(6)

    action_chosen = ACTIONS[np.argmax(predicted_q_values)]
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

def excludeInvalidActions(game_state: dict, q_values_array):
    possibleList = validAction(game_state)
    for i in range(0, len(q_values_array)):
        if not possibleList[i]:
            q_values_array[i] = float(-np.inf)

    return q_values_array

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


def parseBombs(game_state: dict) -> np.array:
    try:
        featurevector = np.zeros((settings.COLS, settings.ROWS))
        for bomb in game_state["bombs"]:
            featurevector[bomb[0][0]][bomb[0][1]] = -bomb[1] - 1
        return featurevector
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
        return np.array([0])

    objective_map = parseField(game_state)
    parseCoins(game_state, objective_map)
    parseSelf(game_state, objective_map)
    parseOthers(game_state, objective_map)

    deadly_map = parseBombs(game_state)
    deadly_map = deadly_map + parseExplosionMap(game_state)

    output_vector = np.append(objective_map.flatten(), deadly_map.flatten())

    return output_vector
