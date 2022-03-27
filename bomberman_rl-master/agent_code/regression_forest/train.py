from collections import namedtuple, deque
import pickle
from typing import List
import os
import numpy as np
import events as e
from .callbacks import state_to_features
from .callbacks import possible_steps
from sklearn.ensemble import RandomForestRegressor
from .callbacks import parse_danger

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 90
# Events
GOLDSEARCH = "GOLDSEARCH"
ENEMYINLINE = "ENEMYINLINE"
BOMBTHREAD = "BOMBTHREAT"
GOLDRUSH = "GOLDRUSH"
WRONGWAY = "WRONGWAY"
WRONGLINE = "WRONGLINE"


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 70,
        e.KILLED_OPPONENT: 150,
        e.GOT_KILLED: -150,
        e.KILLED_SELF: -200,
        e.WAITED: -5,
        e.COIN_FOUND: 8,
        e.CRATE_DESTROYED: 5,
        e.INVALID_ACTION: -15,
        e.BOMB_DROPPED: -1.5,
        e.SURVIVED_ROUND: 1,
        e.BOMBTHREAD: -20,
        e.GOLDSEARCH: 4,
        e.ENEMYINLINE: 6,
        # e.GOLDRUSH: 6,
        e.WRONGWAY: -10,
        # e.WRONGLINE: -3,
        e.MOVED_LEFT: -1.5,
        e.MOVED_RIGHT: -1.5,
        e.MOVED_UP: -1.5,
        e.MOVED_DOWN: -1.5
    }
    # coin = False
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            if e.COIN_COLLECTED in events and (event == e.WRONGWAY or event == e.WRONGLINE):
                continue
            reward_sum += game_rewards[event]
            # if event == e.WRONGWAY:
            # coin = True
    if len(self.transitions) > 3:
       if self.transitions[-3][1] == self.transitions[-1][1] and self.transitions[-2][1] != self.transitions[-1][1]:
           reward_sum -= 5
           self.logger.info(f"Punished repeated action")
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    if reward_sum is None:
        return 0
    return reward_sum


def setup_training(self):
    """
    Initialise self for training purpose.
    This is called after `setup` in callbacks.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Set up an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.past = False
    with open("rewards.pt", "wb") as file:
        pickle.dump([], file)
    self.LEARN_RATE = 0.7
    self.re_overview = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Idea: Add your own events to hand out rewards

    self.oldstate = old_game_state
    self.newstate = new_game_state



    # state_to_features is defined in callbacks.py

    if old_game_state is not None and new_game_state is not None and self_action is not None:
        old = state_to_features(old_game_state)
        new = state_to_features(new_game_state)
        if way_to_go(self.transitions, old):
            events.append(GOLDSEARCH)
        else:
            events.append(WRONGWAY)
        if face_opponent(self.transitions, old):
            events.append(ENEMYINLINE)
        if in_bombs_way(self.oldstate):
            events.append(BOMBTHREAD)
        if way_to_go2(old):
            events.append(GOLDRUSH)
        if not way_to_go2(old):
            events.append(WRONGLINE)
        self.transitions.append(Transition(old, self_action, new, reward_from_events(self, events=events)))

    self.re_overview.append(reward_from_events(self, events))

    if (new_game_state["step"] + 1) % TRANSITION_HISTORY_SIZE == 0:
        y_t = construct_Y(self)
        transition_list_to_data(self, Ys=y_t)
        update_model(self)
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
    self.oldstate = last_game_state
    self.newstate = None
    if last_game_state is not None:
        if last_action is None:
            last_action = "WAIT"
        old = state_to_features(last_game_state)
        if way_to_go(self.transitions, old):
            events.append(GOLDSEARCH)
        else:
            events.append(WRONGWAY)
        if face_opponent(self.transitions, old):
            events.append(ENEMYINLINE)
        if in_bombs_way(self.oldstate):
            events.append(BOMBTHREAD)
        if way_to_go2(old):
            events.append(GOLDRUSH)
        if not way_to_go2(old):
            events.append(WRONGLINE)
        self.transitions.append(Transition(old, last_action, None, reward_from_events(self, events=events)))
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    self.re_overview.append(reward_from_events(self, events))
    # Store the model
    y_t = construct_Y(self)
    transition_list_to_data(self, Ys=y_t)
    update_model(self)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    uni = np.concatenate((self.data, self.rewards.reshape(self.rewards.size, 1)), axis=1)
    uni = np.unique(uni, axis=0)
    self.data = uni[:, :-1]
    self.rewards = uni[:, -1]
    print(f"self.data.shape: {self.data.shape}")
    assert self.rewards.size == self.data.shape[0]
    with open("my-saved-data.pt", "wb") as file:
        pickle.dump(self.data, file)
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
    with open("my-saved-rewards.pt", "wb") as file:
        pickle.dump(self.rewards, file)
    # self.LEARN_RATE = np.exp((-last_game_state["round"] + 1)/500) * 0.8 + 80
    # print(f"Gamma: {self.LEARN_RATE}")
    print(f"Epsilon:{self.random_prob}")

    with open("rewards.pt", "rb") as file:
        re = pickle.load(file)
    re.append(np.mean(self.re_overview))  # if needed: change to measure_performance instead of reward_from_events
    with open("rewards.pt", "wb") as file:
        pickle.dump(re, file)
    self.re_overview = []


def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return ui

def update_model(self):
    y_old = self.data[:, -1].copy()
    self.model = MyModel()
    self.model.train(self.data.copy())
    update_y(self)  # must change
    y_new = self.data[:, -1].copy()
    diff = np.linalg.norm(y_old - y_new)
    print(f"Difference in Y- Calculation: {diff}")
    # if self.oldstate["round"] % 1000 == 0:
    # print(f"Importance of features: {self.model.forest.feature_importances_}")
    if diff <= 1:
        print("no changes")
        raise Exception


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
    if length == 1:
        return 0
    array = np.empty((length, self.transitions[1][0].size + 2))
    array[:, -1] = Ys
    rewards = np.empty(length)
    for i, transition in enumerate(self.transitions):
        if transition[1] is None or transition[0] is None:
            continue
        array[i, :-2] = transition[0]
        array[i, -2] = ACTIONS.index(transition[1])
        # array[i,:-1] = make_dependencies(array[i, :-1], action=ACTIONS.index(transition[1]))
        rewards[i] = transition[-1]
    if np.isnan(array).any():
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
        if transition[2] is None or len(self.transitions) < TRANSITION_HISTORY_SIZE:
            Y.append(transition[-1])
            continue
        elif self.model:
            data = np.empty((1, transition[2].size + 1))
            data[0, :-1] = transition[2]
            ret = []
            for i in possible_steps(self.newstate):
                data[0, -1] = i
                # ret.append(self.model.predict(make_dependencies(data, i)))
                ret.append(self.model.predict(data))
            val = np.max(ret)
        else:
            val = 0
        Y.append(transition[-1] + self.LEARN_RATE * val)
    return Y


class MyModel:
    def __init__(self):
        self.forest = RandomForestRegressor(bootstrap=True, n_estimators=30)
        self.batchsize = 130000
        self.trained = False

    def predict(self, instance):
        return self.forest.predict(instance)

    def train(self, data):
        self.forest = RandomForestRegressor(bootstrap=True, n_estimators=30)
        if data.shape[0] >= self.batchsize:
            data = data[np.random.choice(data.shape[0], self.batchsize), :]
        try:
            self.forest.fit(X=data[:, :-1], y=data[:, -1])
        except:
            print(data[np.argmax(data, axis=0),:])


def in_bombs_way(next):
    s0, s1 = next["self"][3]
    if (parse_danger(next, s0, s1) != 0).any():
        return True
    else:
        return False


def way_to_go(transitions, old):
    if len(transitions) < 2:
        return False
    coins_next = old[0, -2:]
    coins_last = transitions[-1][0][0, -2:]
    coins_last = coins_last.reshape(1, 2)
    coins_next = coins_next.reshape(1, 2)
    if np.sum(np.linalg.norm(coins_last, axis=1)[0]) > np.sum(np.linalg.norm(coins_next, axis=1)[0]):
        return True
    else:
        return False


def way_to_go2(old_state):
    if (old_state[0, -2] == 0 and np.abs(old_state[0, -2]) < 5) or (
            old_state[0, -1] == 0 and np.abs(old_state[0, -2]) < 5):
        return True
    else:
        return False


def face_opponent(transitions, old):
    if len(transitions) < 2:
        return False
    opp_next = old[0, 9:11]
    opp_last = transitions[-1][0][0, 9:11]
    opp_last = opp_last.reshape(1, 2)
    opp_next = opp_next.reshape(1, 2)
    if np.sum(np.linalg.norm(opp_last, axis=1)[0]) > np.sum(np.linalg.norm(opp_next, axis=1)[0]):
        return True
    else:
        return False
