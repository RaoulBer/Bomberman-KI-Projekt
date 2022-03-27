import os
import pickle
import random
from collections import namedtuple, deque
import numpy as np
from . import rolemodel as rm
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
    self.random_prob = 0.90

    if self.train:
        np.random.seed()
        # Fixed length FIFO queues to avoid repeating the same actions
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        # While this timer is positive, agent will not hunt/attack opponents
        self.ignore_others_timer = 0
        self.current_round = 0

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
        self.random_prob = np.exp(-(game_state["round"] + 1)/700)*0.8 + 0.05

    #possible = possible_steps(feature=game_state_use, bomb=game_state['self'][2])
    possible = possible_steps(game_state)
    if (self.train and random.random() < self.random_prob) or not self.model:
        return rm.act(self, game_state)
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
    features = parse_field_orientation(game_state, s0, s1)
    #bombs = parse_bombspots(game_state, s0, s1)
    players = parse_players(game_state, s0, s1)
    coins = parse_coins(game_state, s0, s1)
    danger = parse_danger(game_state, s0, s1)
    danger[(danger == 0).nonzero()] = features[(danger == 0).nonzero()]
    #crate = parse_crate(game_state, s0, s1)
    #np.array([s0, s1]).reshape(1, 2),
    out = np.concatenate((danger, players, coins), axis=1)
    assert out.shape[0] == 1
    return out


def parse_field_orientation(game_state: dict, s0, s1) -> np.array:
    features = np.zeros((1, 9))
    field = game_state["field"]
    mask1 = np.zeros(shape=field.shape, dtype=bool)
    mask1[(s0 - 1):(s0 + 2), (s1 - 1):(s1 + 2)] = True
    field = field[mask1.nonzero()]
    features[0, 0:field.size] = field.flatten()
    features[(features == -1).nonzero()] = 3
    assert features.shape == (1, 9)
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
    players = np.zeros((3, 2))
    others = sorted(game_state["others"], key=lambda x: (s0 - x[-1][0]) ** 2 + (s1 - x[-1][1]) ** 2)
    if others is not None:
        for i, other in enumerate(others):
            if other is None:
                break
            players[i, :] = np.array(
                [np.clip(s0 - other[-1][0], -4, 4), np.clip(s1 - other[-1][1], -4, 4)])
                #[s0 - other[-1][0], s1 - other[-1][1],  99, 99, 1])
    return players.reshape(1, 6)


def parse_coins(game_state, s0, s1) -> np.array:
    coins = np.zeros((1, 2))
    c = sorted(game_state["coins"], key=lambda x: (s0 - x[0]) ** 2 + (s1 - x[1]) ** 2)
    if c:
        for i, coin in enumerate(c):
            if coin is None or i == 1:
                break
            coins[i, :] = np.array([np.clip(s0 - coin[0], -2, 2), np.clip(s1 - coin[1], -2,2)])
    elif (game_state["field"] == 1).any():
        return parse_crate(game_state, s0, s1)
    return coins.reshape(1, 2)

def parse_danger(game_state, s0, s1):
    out = np.zeros((1, 9))
    code_bomb = 2
    code_dead = 3
    b = sorted(game_state["bombs"], key=lambda x: (s0 - x[0][0]) ** 2 + (s1 - x[0][1]) ** 2)
    if b is None:
        return out
    for bomb in b:
        bombx, bomby = bomb[0]
        # 0,0
        if (s0 -1  - bombx == 0 and s1 - 1 - bomby <= 3) or (s1 - 1 - bomby == 0 and s0 - 1 - bombx <= 3):
            out[0,0] = code_bomb
        #0,1
        if (s0 -1- bombx == 0 and s1 - bomby <= 3) or (s1 - bomby == 0 and s0-1 - bombx <= 3):
            out[0, 1] = code_bomb
        #0,-1
        if (s0-1 - bombx == 0 and s1 +1 - bomby <= 3) or (s1 +1 - bomby == 0 and s0-1 - bombx <= 3):
            out[0,2] = code_bomb
        #1,0
        if (s0 - bombx == 0 and s1 - 1 - bomby == 0) or (s0 - bomb[0][0] <= 3 and s1 - 1 - bomb[0][1] <= 3):
            out[0, 3] = code_bomb
        #1,1
        if (s0 - bombx == 0 and s1 - bomb[0][1] <= 3) or (s1 - bomby == 0 and s0 - bomb[0][0] <= 3):
            out[0,4] =code_bomb
        #1,-1
        if (s0 - bombx == 0 and s1 + 1 - bomby <= 3) or (s1 - bomby == 0 and s0 + 1 - bombx <= 3):
            out[0, 5] = code_bomb
        #-1,0
        if (s0 + 1 - bombx == 0 and s1-1 - bomby <= 3) or (s1 -1- bomby == 0 and s0 + 1 - 1 - bombx <= 3):
            out[0,6] = code_bomb
        #-1,1
        if (s0 + 1 - bombx == 0 and s1 - bomby == 0) or (s0 + 1 - bomb[0][0] <= 3 and s1 - bomb[0][1] <= 3):
            out[0, 7] = code_bomb
        #-1,-1
        if (s0 + 1 - bombx == 0 and s1 + 1 - bomb[0][1] <= 3) or (s1 + 1 - bomby == 0 and s0 + 1 - bomb[0][0] <= 3):
            out[0,8] = code_bomb

    if game_state["explosion_map"][s0-1, s1-1] != 0:
            out[0, 0] = code_dead
    if game_state["explosion_map"][s0-1, s1] != 0:
            out[0, 1] = code_dead
    if game_state["explosion_map"][s0-1, s1 + 1] != 0:
            out[0, 2] = code_dead
    if game_state["explosion_map"][s0, s1 - 1] != 0:
            out[0, 3] = code_dead
    if game_state["explosion_map"][s0, s1] != 0:
            out[0, 4] = code_dead
    if game_state["explosion_map"][s0, s1+1] != 0:
            out[0, 5] = code_dead
    if game_state["explosion_map"][s0+1, s1-1] != 0:
            out[0, 6] = code_dead
    if game_state["explosion_map"][s0+1, s1] != 0:
            out[0, 7] = code_dead
    if game_state["explosion_map"][s0+1, s1 + 1] != 0:
            out[0, 8] = code_dead
    return out

def parse_crate(game_state, s0, s1):
    crates = (game_state["field"] == 1).nonzero()
    c0 = crates[0] - s0
    c1 = crates[1] - s1
    crates = np.linalg.norm(np.stack((c0, c1), axis=0), axis=0)
    assert crates.size == c1.size
    #if not crates:
    #    return np.array([0,0]).reshape(1, 2)
    index = np.argmin(crates)
    return np.array([np.clip(c0[index],-2, 2), np.clip(c1[index], -2, 2)]).reshape(1, 2)


def possible_steps(game_state: dict):
    return np.array([0, 1, 2, 3, 4, 5])
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
