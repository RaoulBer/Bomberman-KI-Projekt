from collections import namedtuple, deque
import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features
from .callbacks import possible_steps
from sklearn.ensemble import RandomForestRegressor
from .callbacks import parse_danger
from .callbacks import parse_coins

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 900
# Events
GOLDSEARCH = "GOLDSEARCH"
ENEMYINLINE = "ENEMYINLINE"
BOMBTHREAD = "BOMBTHREAT"
GOLDRUSH = "GOLDRUSH"
WRONGWAY = "WRONGWAY"
WRONGLINE = "WRONGLINE"


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 70, # old_state --> rewarded automatically
        e.KILLED_OPPONENT: 150,  # old_state --> rewarded automatically
        e.GOT_KILLED: -150, # old_state --> rewarded automatically
        e.KILLED_SELF: -100,
        e.WAITED: -5, # old_state --> rewarded automatically
        e.COIN_FOUND: 8, # old_state --> rewarded automatically
        e.CRATE_DESTROYED: 15, # old_state --> rewarded automatically
        e.INVALID_ACTION: -15, # old_state --> rewarded automatically
        e.BOMB_DROPPED: -3, # ? new state --> rewarded automatically !!!
        e.SURVIVED_ROUND: 1, # old_state, small reward anyways
        e.BOMBTHREAD: -20, # old state - been in danger?
        e.GOLDSEARCH: 4, # acient state + old state
        e.ENEMYINLINE: 6, # old state
        e.IN_CIRCLES: -20, # automated
        e.WRONGWAY: -10, # acient state + old state --> NOT Goldsearch
        # e.WRONGLINE: -3,
        e.MOVED_LEFT: -3,
        e.MOVED_RIGHT: -3,
        e.MOVED_UP: -3,
        e.MOVED_DOWN: -3
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
    This is called after `setup` in callbacks_TRAIN.py.
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Set up an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.past = False
    with open("rewards.pt", "wb") as file:
        pickle.dump([], file)
    with open("Y_Convergence.pt", "wb") as file:
        pickle.dump([], file)
    with open("steps.pt", "wb") as file:
        pickle.dump([], file)
    self.LEARN_RATE = 0.99
    self.re_overview = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # Idea: Add your own events to hand out rewards
    # state_to_features is defined in callbacks_TRAIN.py

    if old_game_state is not None and new_game_state is not None and self_action is not None:
        old = state_to_features(old_game_state)
        new = state_to_features(new_game_state)
        if way_to_go(old_game_state, new_game_state):
            events.append(GOLDSEARCH)
        else:
            events.append(WRONGWAY)
        if face_opponent(new, old):
            events.append(ENEMYINLINE)
        if in_bombs_way(new_game_state):
            events.append(BOMBTHREAD)
        self.transitions.append(Transition(old, self_action, new, reward_from_events(self, events=events)))

    self.re_overview.append(reward_from_events(self, events))

    self.oldstate = old_game_state
    #if (new_game_state["step"] + 1) % TRANSITION_HISTORY_SIZE == 0:
    #    y_t = construct_Y(self)
    #    transition_list_to_data(self, Ys=y_t)
    #    update_model(self)
    #    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


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
    #print(last_game_state["step"])
    if last_game_state is not None:
        if last_action is None:
            last_action = "WAIT"
        old = state_to_features(last_game_state)
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
    #print(f"self.data.shape: {self.data.shape}")
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
    with open("steps.pt", "rb") as file:
        re = pickle.load(file)
    re.append(last_game_state["step"])  # if needed: change to measure_performance instead of reward_from_events
    with open("steps.pt", "wb") as file:
        pickle.dump(re, file)
    self.re_overview = []


def update_model(self):
    self.mod =True
    y_old = self.data[:, -1].copy()
    self.model = MyModel()
    self.model.train(self.data.copy())
    update_y(self)  # must change
    y_new = self.data[:, -1].copy()
    diff = np.linalg.norm(y_old - y_new)
    print(f"Difference in Y- Calculation: {diff/self.data.shape[0]}")
    with open("Y_Convergence.pt", "rb") as file:
        re = pickle.load(file)
    re.append(diff/self.data.shape[0])  # if needed: change to measure_performance instead of reward_from_events
    with open("Y_Convergence.pt", "wb") as file:
        pickle.dump(re, file)
    # if self.oldstate["round"] % 1000 == 0:
    # print(f"Importance of features: {self.model.forest.feature_importances_}")
    if diff <= 1:
        print("no changes")
        raise Exception


def update_y(self):
    Y = calculate_y(self)
    self.data[:-1, -1] = Y


def calculate_y(self):
    proposition = self.model.predict(self.data[1:, :-1])
    y = self.rewards[:-1] + self.LEARN_RATE*proposition
    assert y.shape == self.data[:-1, -1].shape
    return y


def transition_list_to_data(self, Ys):
    length = len(self.transitions)
    assert len(Ys) == length -1
    if length == 1:
        return 0
    array = np.zeros((length, self.transitions[1][0].size + 2))
    array[:-1, -1] = Ys
    assert array[0, -1] == Ys[0]
    rewards = np.zeros(length)
    for i, transition in enumerate(self.transitions):
        if i == len(self.transitions) - 1:
            break
        if transition[1] is None or transition[0] is None:
            continue
        array[i, :-2] = transition[0]
        array[i, -2] = ACTIONS.index(transition[1])
        rewards[i] = self.transitions[i-1][-1]
    rewards[-1] += self.transitions[-1][-1]
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
    for i, transition in enumerate(self.transitions):
        if i == 0:
            continue
        if transition[2] is None:
            Y.append(self.transitions[i-1][-1])
            continue
        elif self.mod:
            data = np.empty((1, transition[2].size + 1))
            data[0, :-1] = transition[2]
            ret = []
            for i in possible_steps(None):
                data[0, -1] = i
                if np.isnan(data).any():
                    continue
                ret.append(self.model.predict(data))
            if ret:
                val = np.max(ret)
            else:
                val = 0
        else:
            val=0
        Y.append(transition[-1] + self.LEARN_RATE * val)
    return Y


class MyModel:
    def __init__(self):
        self.forest = RandomForestRegressor(bootstrap=True, n_estimators=100)
        self.batchsize = 140000
        self.trained = False

    def predict(self, instance):
        return self.forest.predict(instance)

    def train(self, data):
        self.forest = RandomForestRegressor(bootstrap=True, n_estimators=100)
        if data.shape[0] >= self.batchsize:
            data = data[np.random.choice(data.shape[0], self.batchsize), :]
        try:
            self.forest.fit(X=data[:-1, :-1], y=data[:-1, -1])
        except:
            print(data[np.argmax(data, axis=0), :])


def in_bombs_way(next):
    s0, s1 = next["self"][3]
    if (parse_danger(next, s0, s1) != 0).any():
        return True
    else:
        return False


def way_to_go(game_state_o, game_state_n):
    if game_state_o is None:
        return False
    s0o, s1o = game_state_o["self"][3]
    coin_o = parse_coins(game_state_o, s0o, s1o)
    s0n, s1n = game_state_n["self"][3]
    coin_n = parse_coins(game_state_n, s0n, s1n)
    assert len(np.linalg.norm(coin_o, axis=1)) == 1
    if np.sum(np.linalg.norm(coin_o, axis=1)[0]) > np.sum(np.linalg.norm(coin_n, axis=1)[0]):
        return True
    else:
        return False


def way_to_go2(game_state):
    if game_state is None:
        return False
    s0, s1 = game_state["self"][3]
    coin = parse_coins(game_state, s0, s1)
    if (coin[0,-2] == 0 and np.abs(coin[0,-1]) < 5) or (
            coin[0,-1] == 0 and np.abs(coin[0, -2]) < 5):
        return True
    else:
        return False


def face_opponent(old, new):
    if old is None or new is None:
        return False
    opp_next = new[0, 9:11]
    opp_last = old[0, 9:11]
    opp_last = opp_last.reshape(1, 2)
    opp_next = opp_next.reshape(1, 2)
    if (opp_last == np.array([0.0, 0.0]).reshape(1,2)).all() or (opp_next==np.array([0.0, 0.0]).reshape(1,2)).all():
        return True
    if np.sum(np.linalg.norm(opp_last, axis=1)[0]) > np.sum(np.linalg.norm(opp_next, axis=1)[0]):
        return True
    else:
        return False