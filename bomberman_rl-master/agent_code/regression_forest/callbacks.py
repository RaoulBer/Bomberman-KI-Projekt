import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
prob = [.2, .2, .2, .2, .1, .1]


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
    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        #with open("saved_feature_reduction.pt", "rb") as file:
        #    self.feature_red = pickle.load(file)


def act(self, game_state: dict) -> str:
    if game_state["step"] == 10 and game_state["round"] == 1:
        with open("game_setup.pt", "wb") as file:
            pickle.dump(game_state, file)
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    self.round = 0
    self.random_prob = 1
    if game_state["round"] > self.round:
        self.round = self.round + 1
        self.random_prob = self.random_prob * 0.95
    game_state_use = state_to_features(game_state)
    possible = possible_steps(feature=game_state_use, game_state=game_state)
    if (self.train and random.random() < self.random_prob) or not os.path.isfile("my-saved-model.pt"):
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice([ACTIONS[i] for i in possible], p=return_distro(possible)) #, p=[prob[i] for i in possible]

    self.logger.debug("Querying model for action.")

    #feature_state = self.feature_red(game_state_use)
    data = np.empty((1, game_state_use.size + 1))
    data[:, :-1] = game_state_use
    max = -1000
    for i in possible:
        data[0, -1] = i
        mod = self.model.predict(data)
        if mod > max:
            max = mod
            response = i
    return ACTIONS[response]


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

    # For example, you could construct several channels of equal shape, ...
    """
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
    """
    features = np.zeros((4+176, 1))
    bombs = np.zeros((4, 3))
    coins = np.zeros((3,2))
    players = np.zeros((4, 3))
    field = game_state["field"]
    mask = np.ones(shape=field.shape, dtype=bool)
    mask[field == -1] = False
    expl = game_state["explosion_map"]
    expl = expl[(mask).nonzero()]
    *_, (s0, s1) = game_state["self"]
    features[0] = game_state["round"]
    features[1] = game_state["step"]
    features[2] = s0
    features[3] = s1
    features[4:, 0] = 3 * expl.flatten()
    b = game_state["bombs"].sort(key=lambda x: x[0][0]**2 + x[0][1]**2)
    others = game_state["others"].sort(key=lambda x: x[-1][0]**2 + x[-1][1]**2)
    if b is not None:
        for i, bomb in enumerate(b):
            bombs[i,:] = np.array(s0-[bomb[0][0], s1-bomb[0][1], bomb[1]])
    if others is not None:
        for i, other in enumerate(others):
            if other is None:
                break
            if other[2]:
                players[i, :] = np.array(s0-[other[-1][0], s1-other[-1][1], -5])
            else:
                players[i, :] = np.array(s0-[other[-1][0], s1-other[-1][1], other[1]])

    c = game_state["coins"].sort(key=lambda x: x[0]**2 + x[1]**2)
    if c is not None:
        for i, coin in enumerate(c):
            if coin is None:
                break
            coins[i, :] = np.array(s0-[coin[0], s1-coin[1]])

    bombs = bombs.reshape(12, 1)
    players = players.reshape(12, 1)
    coins = coins.reshape(6, 1)
    out = np.concatenate((features, players, bombs, coins), axis=0).T
    assert out.shape[0] == 1
    return out

def possible_steps(game_state, feature):
    actions = [4]
    j=3
    i=2
    if feature[0, i] % 2 == 1:
        if feature[0,j] < 15:
            if feature[0,j] > 1:
                actions.append(0)
                actions.append(2)
            else:
                actions.append(2)
        else:
            actions.append(0)
    if feature[0,j] % 2 == 1:
        if feature[0,i] < 15:
            if feature[0,i] >1:
                 actions.append(1)
                 actions.append(3)
            else:
                actions.append(1)
        else:
            actions.append(3)
    if game_state['self'][2]:
        actions.append(5)
    return np.sort(actions)

def return_distro(actions):
    length = len(actions)
    if 5 in actions:
        frac = 1/(length-1)
    else:
        frac = 1/(length-0.5)
    out = np.array([frac for _ in range(length)])
    out[actions > 3] = frac/2
    return out
