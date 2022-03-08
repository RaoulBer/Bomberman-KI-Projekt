import os
import pickle
import random

import keras.models
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

import settings

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#model parameters
action_size = len(ACTIONS)
gamma = 0.95
learning_rate = 0.001

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
coinsNum = 4
selfNum = 3 # bomb is possible plus x and y coordinate
othersNum = 3 * 3 # bomb is possible plus x and y coordinate times 3
featureSum = roundNum + stepNum + fieldNum + bombsNum + explosion_mapNum + coinsNum + selfNum + othersNum

def build_model():
    model = Sequential()
    model.add(Dense(featureSum, input_dim=featureSum, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return model

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
    if self.train or not os.path.isfile("my-saved-model"):
        self.logger.info("Setting up model from scratch.")
        self.model = build_model()
    else:
        self.logger.info("Loading model from saved state.")
        #with open("my-saved-model.pt", "rb") as file:
            #self.model = pickle.load(file)
        self.model = keras.models.load_model("my-saved-model")

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
    return ACTIONS[np.argmax(self.model.predict(state_to_features(game_state)))]

def parseBombs(game_state: dict) -> np.array:
    bombVector = []
    for bomb in game_state["bombs"]:
        bombVector.append(bomb[0][0])
        bombVector.append(bomb[0][1])
        bombVector.append(bomb[1])

    while len(bombVector) < bombsNum:
        bombVector.append(0)

    return np.array(bombVector)

def parseCoins(game_state: dict) ->np.array:
    coinVector = []
    for coin in game_state["coins"]:
        coinVector.append(coin[0])
        coinVector.append(coin[1])

    while len(coinVector) < coinsNum:
        coinVector.append(0)

    return np.array(coinVector)

def parseOthers(game_state: dict) ->np.array:
    otherVector = []
    for other in game_state["others"]:
        otherVector.append(other[2])
        otherVector.append(other[3][0])
        otherVector.append(other[3][1])

    while len(otherVector) < othersNum:
        otherVector.append(0)

    return np.array(otherVector)

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
    featurevector[0] = game_state["round"]
    temp += 1
    featurevector[temp] = game_state["step"]
    temp += 1
    featurevector[temp:temp+fieldNum] = game_state["field"].reshape(-1)
    temp += fieldNum
    featurevector[temp:temp+bombsNum] = parseBombs(game_state)
    temp += bombsNum
    featurevector[temp:temp+explosion_mapNum] = game_state["explosion_map"].reshape(-1)
    temp += explosion_mapNum
    featurevector[temp:temp+coinsNum] = parseCoins(game_state)
    temp += coinsNum
    featurevector[temp] = game_state["self"][2]
    temp += 1
    featurevector[temp] = game_state["self"][3][0]
    temp += 1
    featurevector[temp] = game_state["self"][3][1]
    temp += 1
    featurevector[temp:temp+othersNum] = parseOthers(game_state)

    return np.array([featurevector])
