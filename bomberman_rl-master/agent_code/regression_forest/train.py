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


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 10  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARN_RATE = 0.1

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
    if os.path.isfile("my-saved-model.pt"):
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    with open("rewards.pt", "wb") as file:
        pickle.dump([], file)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.
    This is *one* of the places where you could update your agent.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
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

    with open("my-saved-data.pt", "rb") as file:
        array = pickle.load(file)

    output_data = unique(array)

    with open("my-saved-data.pt", "wb") as file:
        pickle.dump(output_data, file)


    update_model(self)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

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
    with open("my-saved-data.pt", "rb") as file:
        data = pickle.load(file)
    self.reduction = PCA(n_components=40)
    if data.shape[0] > 40:
        data_set = np.column_stack((self.reduction.fit_transform(data[:, :-2]), data[:, -2], data[:,:-1]))
        with open("dimension_reduction.pt", "wb") as file:
            pickle.dump(self.reduction, file)
        with open("my-saved-data.pt", "wb") as file:
            pickle.dump(data_set, file)
    else:
        data_set = data

    if data_set.shape[0] >=800:
        data_set = data_set[np.random.choice(data_set.shape[0], 800), :]
    self.logger.info(f"Trained with a data-set of this size: {data_set.shape}")
    self.model = RandomForestRegressor(max_depth=300, random_state=0)
    self.model.fit(data_set[:, :-1], data_set[:, -1])
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


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
    array = np.nan_to_num(array, copy=True)
    if not os.path.isfile("my-saved-data.pt"):
        with open("my-saved-data.pt", "wb") as file:
            pickle.dump(array, file)
    else:
        with open("my-saved-data.pt", "rb") as file:
            data_set = pickle.load(file)
        data_set = np.concatenate((data_set, array))
        with open("my-saved-data.pt", "wb") as file:
            pickle.dump(data_set, file)
    return array

def construct_Y(self):
    Y = []
    for transition in self.transitions:
        if transition[2] is None:
            Y.append(transition[-1])
            continue
        elif type(self.model) != type(np.empty(1)) and transition[0] is not None:
            data = np.empty((1, transition[2].size + 1))
            data[:, :-1] = transition[0]
            data[0, -1] = ACTIONS.index(transition[1])
            data2 = np.empty((1, transition[2].size + 1))
            data2[0,:-1] = transition[2]
            val = self.model.predict(data)
            ret = []
            for i in possible_steps(feature = transition[2], bomb = True):
                data2[0, -1] = i
                ret.append(self.model.predict(data2))
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

