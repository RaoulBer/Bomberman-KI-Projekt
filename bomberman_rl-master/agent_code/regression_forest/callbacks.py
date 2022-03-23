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
    self.round = 1
    self.random_prob = 0.9

    if self.train:
        self.model = False
        self.rewards = False
        if os.path.isfile("my-saved-data.pt"):
            with open("my-saved-data.pt", "rb") as file:
                self.data = pickle.load(file)
            with open("my-saved-rewards.pt", "rb") as file:
                self.rewards = pickle.load(file)
        else:
            self.data = False
    else:
        with open("my-saved-model.pt", "rb") as file:
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
    if game_state["round"] != self.round:
        self.round = game_state["round"]
        self.random_prob = np.exp(-(game_state["round"] + 1)/700)*0.8 + 0.15

    #possible = possible_steps(feature=game_state_use, bomb=game_state['self'][2])
    possible = possible_steps(game_state)
    if (self.train and random.random() < self.random_prob) or not self.model:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice([ACTIONS[i] for i in possible], p=return_distro(possible))

    self.logger.debug("Querying model for action.")
    game_state_use = state_to_features(game_state)

    highest_known = -1000
    response = 4  # Standard: Wait
    for i in possible:
        data = np.empty((1, game_state_use.size + 1))
        data[0, :-1] = game_state_use
        data[0, -1] = i
        #data = make_dependencies(data, i)
        mod = self.model.predict(data)
        if mod > highest_known:
            highest_known = mod
            response = i
    return ACTIONS[response]


def state_to_features(game_state: dict) -> np.array:
    """
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        print("game state is none")
        return None

    s0, s1 = game_state["self"][3]
    #features = parse_field_orientation(game_state, s0, s1)
    #bombs = parse_bombspots(game_state, s0, s1)
    players = parse_players(game_state, s0, s1)
    coins = parse_coins(game_state, s0, s1)
    danger = parse_danger(game_state, s0, s1)
    crate = parse_crate(game_state, s0, s1)
    out = np.concatenate((danger, players, crate, coins), axis=1)
    assert out.shape[0] == 1
    return out


def parse_field_orientation(game_state: dict, s0, s1) -> np.array:
    features = np.zeros((1, 4+25))
    features[0, 0] = game_state["round"]
    features[0, 1] = np.floor(game_state["step"] / 100)
    features[0, 2] = s0
    features[0, 3] = s1
    #features[0, 4] = 99
    #features[0, 5] = 99
    field = game_state["field"]
    mask2 = np.ones(shape=field.shape, dtype=bool)
    mask1 = np.zeros(shape=field.shape, dtype=bool)
    mask1[np.max([0, s0 - 2]):np.min([16, s0 + 2]), np.max([0, s1 - 2]):np.min([16, s1 + 2])] = True
    mask2[field == -1] = False
    mask2 = mask2[mask1.nonzero()]
    expl_crates = game_state["explosion_map"][mask1.nonzero()]
    field = field[mask1.nonzero()]
    expl_crates = - expl_crates[mask2.nonzero()] + field[mask2.nonzero()]
    features[0, 4:expl_crates.size + 4] = 3 * expl_crates.flatten()

    assert features.shape == (1, 52)
    return features


def parse_bombspots(game_state: dict, s0, s1) -> np.array:
    bombs = np.zeros((4, 3))
    b = sorted(game_state["bombs"], key=lambda x: (s0 - x[0][0]) ** 2 + (s1 - x[0][1]) ** 2)
    if b is not None:
        for i, bomb in enumerate(b):
            bombx, bomby = bomb[0]
            bombs[i, :] = np.array([s0 - bombx, s1 - bomby, bomb[1]])
            #bombs[i, :] = np.array([s0 - bomb[0][0], s1 - bomb[0][1], 99, 99, bomb[1]])
    return bombs.reshape(1, 12)


def parse_players(game_state: dict, s0, s1) -> np.array:
    players = np.zeros((3, 3))
    others = sorted(game_state["others"], key=lambda x: (s0 - x[-1][0]) ** 2 + (s1 - x[-1][1]) ** 2)
    if others is not None:
        for i, other in enumerate(others):
            if other is None:
                break
            if other[2]:
                players[i, :] = np.array(
                    #[s0 - other[-1][0], s1 - other[-1][1],  99, 99, -1])
                    [np.sign(s0 - other[-1][0]), np.sign(s1 - other[-1][1]), -1])
            else:
                players[i, :] = np.array(
                    [np.sign(s0 - other[-1][0]), np.sign(s1 - other[-1][1]), 1])
                    #[s0 - other[-1][0], s1 - other[-1][1],  99, 99, 1])
    return players.reshape(1, 9)


def parse_coins(game_state, s0, s1) -> np.array:
    coins = np.zeros((1, 2))
    c = sorted(game_state["coins"], key=lambda x: (s0 - x[0]) ** 2 + (s1 - x[1]) ** 2)
    if c is not None:
        for i, coin in enumerate(c):
            if coin is None or i == 1:
                break
            coins[i, :] = np.array([s0 - coin[0], s1 - coin[1]])
    return coins.reshape(1, 2)

def parse_danger(game_state, s0, s1):
    out = np.zeros((1, 5))
    b = sorted(game_state["bombs"], key=lambda x: (s0 - x[0][0]) ** 2 + (s1 - x[0][1]) ** 2)
    if b is None:
        return out
    for bomb in b:
        bombx, bomby = bomb[0]
        if (s0 - bombx == 0 and s1 - bomby <= 3) or (s1 - bomby == 0 and s0 - bombx <= 3):
            out[0,0] = -1
        if (s0 + 1 - bombx == 0 and s1 - bomby <= 3) or (s1 - bomby == 0 and s0 + 1 - bombx <= 3):
            out[0, 1] = -1
        if (s0 - 1 - bombx == 0 and s1 - bomby <= 3) or (s1 - bomby == 0 and s0 - 1 - bombx <= 3):
            out[0,2] = -1
        if (s0 - bombx == 0 or s1 + 1 - bomby == 0) and (s0 - bomb[0][0] <= 3 or s1 + 1 - bomb[0][1] <= 3):
            out[0, 3] = -1
        if (s0 - bombx == 0 and s1 + 1 - bomb[0][1] <= 3)  or (s1 + 1 - bomby == 0 and s0 - bomb[0][0] <= 3):
            out[0,4] = -1
    if game_state["explosion_map"][s0, s1] != 0:
            out[0, 0] = -2
    if game_state["explosion_map"][s0+1, s1] != 0:
            out[0, 1] = -2
    if game_state["explosion_map"][s0 -1, s1] != 0:
            out[0, 2] = -2
    if game_state["explosion_map"][s0, s1 +1] != 0:
            out[0, 3] = -2
    if game_state["explosion_map"][s0, s1 -1] != 0:
            out[0, 4] = -2
    return out

def parse_crate(game_state, s0, s1):
    crates = (game_state["field"] == 1).nonzero()
    c0 = crates[0] - s0
    c1 = crates[1] - s1
    crates = np.linalg.norm(np.stack((c0, c1), axis=0), axis=0)
    assert crates.size == c1.size
    index = np.argmin(crates)
    return np.array([np.sign(c0[index]), np.sign(c1[index])]).reshape(1,2)


def possible_steps(game_state: dict):
    validAction = [4]
    if game_state is None:
        return np.array([0, 1, 2, 3, 4, 5])
    playerx, playery = game_state["self"][3]
    #UP
    if (game_state["field"][playerx][playery-1] == 0):
        validAction.append(0)
    #RIGHT
    if (game_state["field"][playerx+1][playery] == 0):
        validAction.append(1)
    #DOWN
    if (game_state["field"][playerx][playery+1] == 0):
        validAction.append(2)
    #LEFT
    if (game_state["field"][playerx-1][playery] == 0):
        validAction.append(3)

    if game_state["self"][2]:
        validAction.append(5)
    return np.sort(validAction)


def return_distro(actions):
    length = len(actions)
    if 5 in actions:
        frac = 1 / (length - 1)
    else:
        frac = 1 / (length - 0.5)
    out = np.array([frac for _ in range(length)])
    out[actions > 3] = frac / 2
    return out


def make_dependencies(data, action):
    if len(data.shape) == 2:
        places, take_from = return_doubles1(data)
        data[0, places] = data[0, [take_from]] * action
    else:
        places, take_from = return_doubles2(data)
        data[places] = data[take_from] * action
    return data


def return_doubles1(feature):
    places = (feature == 99).nonzero()[1]
    take_from = places - 2
    return places, take_from


def return_doubles2(feature):
    places = (feature == 99).nonzero()[0]
    take_from = places - 2
    return places, take_from
