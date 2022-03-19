from collections import namedtuple, deque
import os
import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
from .callbacks import possible_steps
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from .callbacks import make_dependencies


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 15  # keep only ... last transitions
#RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARN_RATE = 0.4

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    with open("rewards.pt", "wb") as file:
        pickle.dump([], file)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if False:
        events.append(PLACEHOLDER_EVENT)

    with open("rewards.pt", "rb") as file:
        re = pickle.load(file)
    re.append(measure_performance(self, events))
    with open("rewards.pt", "wb") as file:
        pickle.dump(re, file)

    # state_to_features is defined in callbacks.py
    if old_game_state is not None and new_game_state is not None and self_action is not None:
        self.transitions.append(
            Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state),
                       reward_from_events(self, events)))
    if new_game_state["step"] + 1 % TRANSITION_HISTORY_SIZE == 0:
        y_t = construct_Y(self)
        transition_list_to_data(self, Ys=y_t)
        update_model(self, weights=None)
        self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.
    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.
    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.
    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    if last_game_state is not None and last_action is not None:
        self.transitions.append(
            Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    # todo update/write new model here :)
    y_t = construct_Y(self)
    transition_list_to_data(self, Ys=y_t)
    self.data = unique(self.data)
    update_model(self)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    with open("my-saved-data.pt", "wb") as file:
        pickle.dump(self.data, file)
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 5,
        e.GOT_KILLED: -10,
        e.KILLED_SELF: -8,
        e.WAITED: -2,
        e.CRATE_DESTROYED: 3,
        e.INVALID_ACTION: -2,
        e.BOMB_DROPPED: -0.5
        #e.MOVED_LEFT: 1.5,
        #e.MOVED_RIGHT: 1.5,
        #e.MOVED_UP: 1.5,
        #e.MOVED_DOWN: 1.5
        # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #if self.transitions:
    #    reward_sum += self.transitions[-1][-2][:,1] * 0.1
    if len(self.transitions) > 3:
        if self.transitions[-3][1] == self.transitions[-1][1] and self.transitions[-2][1] != self.transitions[-1][1]:
            reward_sum -= 5
            self.logger.info(f"Punished repeated action")
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    if reward_sum is None:
        return 0
    return reward_sum


def update_model(self, weights=None):
    # todo Dimension-reduction nicht speichern in my-saved-data, nur lernen im forest etc. -
    #  dadurch kann noch korrigiert werden falls features sich ändern.
    # todo generate new Y each round
    self.model = MyModel()
    self.model.train(self.data)
    update_y(self)


def update_y(self):
    Y = calculate_y(self)
    self.data[:, -1] = Y


def calculate_y(self):
    proposition = self.model.predict(self.data[:, :-1])
    y = self.rewards + LEARN_RATE * proposition
    assert y.shape == self.data[:, -1].shape
    return y


def transition_list_to_data(self, Ys):
    length = len(self.transitions)
    assert len(Ys) == length
    array = np.empty((length, self.transitions[1][0].size + 2))
    array[:, -1] = Ys
    for i, transition in enumerate(self.transitions):
        if transition[1] is None or transition[0] is None:
            continue
        array[i, :-2] = transition[0]
        array[i, -2] = ACTIONS.index(transition[1])
        array = make_dependencies(array, action=ACTIONS.index(transition[1]))
    array = np.nan_to_num(array, copy=True)
    self.data = np.concatenate((self.data, array))


def construct_Y(self):
    Y = []
    for transition in self.transitions:
        if transition[2] is None:
            Y.append(transition[-1])
            continue
        elif type(self.model) != type(np.empty(1)) and transition[0] is not None:
            data = np.empty((1, transition[2].size + 1))
            data[:, :-1] = make_dependencies(transition[0], ACTIONS.index(transition[1]))
            data[0, -1] = ACTIONS.index(transition[1])
            data2 = np.empty((1, transition[2].size + 1))
            data2[0,:-1] = transition[2]
            #val = self.model.predict(data)
            ret = []
            for i in possible_steps(feature = transition[2], bomb = True):
                data2[0, -1] = i
                if not self.reduce:
                    ret.append(self.model.predict(make_dependencies(data2, i)))
                else:
                    ret.append(self.model.predict(self.reduction.transform(make_dependencies(data2, i))))
            val2 = np.max(ret)
        else:
            val = 0
            val2 = 0
        Y.append(transition[-1] + LEARN_RATE * val2)
    return Y


def measure_performance(self, events: List[str]):
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 6,
        e.GOT_KILLED: -4,
        e.KILLED_SELF: -4,
        e.CRATE_DESTROYED: 1.5,
    }
    reward_sum = -1
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # if self.transitions:
    #    reward_sum += self.transitions[-1][-2][:,1]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    if reward_sum is None:
        return 0
    return reward_sum

class MyModel:
    def __init__(self):
        self.forest = RandomForestRegressor(max_depth=300, random_state=0)
        self.reduction = False

    def predict(self, instance):
        if self.reduction:
            return self.forest.predict(self.reduction.transform(instance))
        else:
            return self.forest.predict(instance)

    def train(self, data):
        try:
            self.forest = RandomForestRegressor(max_depth=300, random_state=0)
            self.reduction = PCA(n_components=40)
            self.reduction.fit(data[:, :-1], y=data[:, -1])
            self.forest.fit(self.reduction.transform(data[:, :-1]), data[:, -1])
        except:
            self.forest = RandomForestRegressor(max_depth=300, random_state=0)
            self.forest.fit(data[:, :-1], data[:, -1])
            self.reduction = False
