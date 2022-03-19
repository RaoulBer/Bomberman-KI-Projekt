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
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
GOLDSEARCH = "GOLDSEARCH"
ENEMYINLINE = "ENEMYINLINE"
BOMBTHREAD = "BOMBTHREAT"
GOLDRUSH = "GOLDRUSH"


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
    self.LEARN_RATE = 0.2


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards

    with open("rewards.pt", "rb") as file:
        re = pickle.load(file)
    re.append(measure_performance(self, events))
    with open("rewards.pt", "wb") as file:
        pickle.dump(re, file)

    # state_to_features is defined in callbacks.py
    if old_game_state is not None and new_game_state is not None and self_action is not None:
        if way_to_go(state_to_features(self, old_game_state), state_to_features(self, new_game_state)):
            events.append(GOLDSEARCH)
        #if face_opponent(state_to_features(self, new_game_state)):
        #    events.append(ENEMYINLINE)
        if in_bombs_way(state_to_features(self, new_game_state)):
            events.append(BOMBTHREAD)
        if way_to_go2(state_to_features(self, new_game_state)):
            events.append(GOLDRUSH)
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
    y_t = construct_Y(self)
    transition_list_to_data(self, Ys=y_t)
    update_model(self)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    with open("my-saved-data.pt", "wb") as file:
        pickle.dump(self.data, file)
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    with open("my-saved-rewards.pt", "wb") as file:
        pickle.dump(self.rewards, file)
    self.LEARN_RATE = self.LEARN_RATE * 0.95 + 0.03



'''
def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]
'''


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
    y = self.rewards + self.LEARN_RATE * proposition
    assert y.shape == self.data[:, -1].shape
    return y


def transition_list_to_data(self, Ys):
    length = len(self.transitions)
    assert len(Ys) == length
    array = np.empty((length, self.transitions[1][0].size + 2))
    array[:, -1] = Ys
    rewards = np.empty(length)
    for i, transition in enumerate(self.transitions):
        if transition[1] is None or transition[0] is None:
            continue
        array[i, :-2] = transition[0]
        array[i, -2] = ACTIONS.index(transition[1])
        array = make_dependencies(array, action=ACTIONS.index(transition[1]))
        rewards[i] = transition[-1]
    array = np.nan_to_num(array, copy=True)
    try:
        self.data = np.concatenate((self.data, array))
    except ValueError:
        self.data = array
    try:
        self.rewards = np.concatenate((self.rewards, rewards))
    except ValueError:
        self.rewards = rewards


def construct_Y(self):
    Y = []
    for transition in self.transitions:
        if transition[2] is None:
            Y.append(transition[-1])
            continue
        elif self.model:
            data = np.empty((1, transition[2].size + 1))
            data[0, :-1] = transition[2]
            ret = []
            for i in possible_steps(feature=transition[2], bomb=True):
                data[0, -1] = i
                ret.append(self.model.predict(make_dependencies(data, i)))
            val = np.max(ret)
        else:
            val = 0
        Y.append(transition[-1] + self.LEARN_RATE * val)
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
        return self.forest.predict(instance)

    def train(self, data):
        self.forest = RandomForestRegressor(max_depth=300, random_state=0)
        if data.shape[0] >= 1600:
            data = data[np.random.choice(data.shape[0], 1600), :]
        self.forest.fit(data[:, :-1], data[:, -1])

        """
        if self.reduction:
            return self.forest.predict(self.reduction.transform(instance))
        else:
         try:
            
            self.forest = RandomForestRegressor(max_depth=300, random_state=0)
            self.reduction = PCA(n_components=40)
            self.reduction.fit(data[:, :-1], y=data[:, -1])
            self.forest.fit(self.reduction.transform(data[:, :-1]), data[:, -1])
        except ValueError:
        """


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 10,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: 3,
        e.WAITED: -1,
        e.CRATE_DESTROYED: 3,
        e.INVALID_ACTION: -2,
        e.BOMB_DROPPED: -0.5,
        e.SURVIVED_ROUND: 5,
        e.BOMBTHREAD: -8,
        e.GOLDSEARCH: 5,
        e.ENEMYINLINE: 1,
        e.GOLDRUSH: 2
        # e.MOVED_LEFT: 1.5,
        # e.MOVED_RIGHT: 1.5,
        # e.MOVED_UP: 1.5,
        # e.MOVED_DOWN: 1.5
        # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # if self.transitions:
    #    reward_sum += self.transitions[-1][-2][:,1] * 0.1
    if len(self.transitions) > 3:
        if self.transitions[-3][1] == self.transitions[-1][1] and self.transitions[-2][1] != self.transitions[-1][1]:
            reward_sum -= 2
            self.logger.info(f"Punished repeated action")
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    if reward_sum is None:
        return 0
    return reward_sum


def in_bombs_way(next):
    if (next[0, 55] == 0 or next[0, 56] == 0) and (next[0, 55] < 3 or next[0, 56] < 3):
        return True
    else:
        return False


def way_to_go(last_state, next_state):
    coins_next = next_state[0, -12::2].reshape(2, 3)
    coins_last = last_state[0, -12::2].reshape(2, 3)
    if np.sum(np.linalg.norm(coins_last, axis=0)[0]) > np.sum(np.linalg.norm(coins_next, axis=0)[0]):
        return True
    else:
        return False


def way_to_go2(next_state):
    if (next_state[0, -12] == 0 or next_state[0, -11] == 0) and (next_state[0, -12] < 3 or next_state[0, -11] < 3) \
            and (next_state[0, -12] != next_state[0, -11]).any() :
        return True
    else:
        return False


def face_opponent(next_state):
    if (next_state[0, 70] == 0 or next_state[0, 71] == 0) and (next_state[0, 70] < 3 or next_state[0, 71] < 3):
        return True
    else:
        return False
