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
    self.random_prob = 1

    if self.train:
        self.model = False
        if os.path.isfile("my-saved-data.pt"):
            with open("my-saved-data.pt") as file:
                self.data = pickle.load(file)
        else:
            self.data = False
    else:
        with open("my-saved-model.pt") as file:
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
        self.random_prob = self.random_prob * 0.88 + 0.01
        print(self.random_prob)

    game_state_use = state_to_features(self, game_state)
    possible = possible_steps(feature=game_state_use, bomb=game_state['self'][2])
    if (self.train and random.random() < self.random_prob) or not self.model:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice([ACTIONS[i] for i in possible], p=return_distro(possible))

    self.logger.debug("Querying model for action.")

    highest_known = -1000
    response = 4  # Standard: Wait
    for i in possible:
        data = np.empty((1, game_state_use.size + 1))
        data[0, :-1] = game_state_use
        data[0, -1] = i
        data = make_dependencies(data, i)
        mod = self.model.predict(data)
        if mod > highest_known:
            highest_known = mod
            response = i
    return ACTIONS[response]


def state_to_features(self, game_state: dict) -> np.array:
    """
    :param self: Stuff
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # todo: Funktionale Programmierung implementieren.

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    *_, (s0, s1) = game_state["self"]
    features = np.zeros((4 + 50, 1))
    bombs = np.zeros((4, 5))
    coins = np.zeros((3, 4))
    players = np.zeros((3, 5))
    features[0] = game_state["round"]
    features[1] = np.floor(game_state["step"] / 100)
    features[2] = s0
    features[3] = s1
    field = game_state["field"]
    mask2 = np.ones(shape=field.shape, dtype=bool)
    mask1 = np.zeros(shape=field.shape, dtype=bool)
    mask1[np.max([0, s0 - 4]):np.min([16, s0 + 4]), np.max([0, s1 - 4]):np.min([16, s1 + 4])] = True
    mask2[field == -1] = False
    mask2 = mask2[mask1.nonzero()]
    expl_crates = game_state["explosion_map"][mask1.nonzero()]
    field = field[mask1.nonzero()]
    expl_crates = - expl_crates[mask2.nonzero()] + field[mask2.nonzero()]
    features[4:expl_crates.size + 4, 0] = 3 * expl_crates.flatten()
    features[-2:, 0] = np.array([s0, s1])
    b = sorted(game_state["bombs"], key=lambda x: (s0 - x[0][0]) ** 2 + (s1 - x[0][1]) ** 2)
    others = sorted(game_state["others"], key=lambda x: (s0 - x[-1][0]) ** 2 + (s1 - x[-1][1]) ** 2)
    if b is not None:
        for i, bomb in enumerate(b):
            bombs[i, :] = np.array([s0 - bomb[0][0], s1 - bomb[0][1], bomb[1], s0 - bomb[0][0], s1 - bomb[0][1]])
            # bombs[i, :] = np.array([s0 - bomb[0][0], s1 - bomb[0][1], bomb[1], 99, 99])
    if others is not None:
        for i, other in enumerate(others):
            if other is None:
                break
            if other[2]:
                players[i, :] = np.array(
                    [s0 - other[-1][0], s1 - other[-1][1], -1, s0 - other[-1][0], s1 - other[-1][1]])
                # players[i, :] = np.array([s0 - other[-1][0], s1 - other[-1][1], -1, 99, 99])
            else:
                players[i, :] = np.array(
                    [s0 - other[-1][0], s1 - other[-1][1], 1, s0 - other[-1][0], s1 - other[-1][1]])
                # players[i, :] = np.array([s0 - other[-1][0], s1 - other[-1][1], 1, 99, 99])

    c = sorted(game_state["coins"], key=lambda x: (s0 - x[0]) ** 2 + (s1 - x[1]) ** 2)
    if c is not None:
        for i, coin in enumerate(c):
            if coin is None or i == 3:
                break
            coins[i, :] = np.array([s0 - coin[0], s1 - coin[1], s0 - coin[0], s1 - coin[1]])
            # coins[i, :] = np.array([s0 - coin[0], s1 - coin[1], 99, 99])

    bombs = bombs.reshape(20, 1)
    players = players.reshape(15, 1)
    coins = coins.reshape(12, 1)
    out = np.concatenate((features, players, bombs, coins), axis=0).T
    assert out.shape[0] == 1
    return out


def possible_steps(feature, bomb=True):
    actions = [4]
    j = 3
    i = 2
    if feature[0, i] % 2 == 1:
        if feature[0, j] < 15:
            if feature[0, j] > 1:
                actions.append(0)
                actions.append(2)
            else:
                actions.append(2)
        else:
            actions.append(0)
    if feature[0, j] % 2 == 1:
        if feature[0, i] < 15:
            if feature[0, i] > 1:
                actions.append(1)
                actions.append(3)
            else:
                actions.append(1)
        else:
            actions.append(3)
    if bomb:
        actions.append(5)
    return np.sort(actions)


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
    for i in range(data.shape[0]):
        data[i, [52, 53, 57, 58, 62, 63, 67, 68, 72, 73, 77, 78, 82, 83, 91, 92, 95, 96, 99, 100]] = \
            data[i, [52, 53, 57, 58, 62, 63, 67, 68, 72, 73, 77, 78, 82, 83, 91, 92, 95, 96, 99, 100]] * action
    return data
