from collections import namedtuple, deque
import os
import pickle
from typing import List
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import events as e
from .callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
LEARN_RATE = 0.6

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

    # state_to_features is defined in callbacks.py
    if old_game_state is not None:
        self.transitions.append(
            Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                       reward_from_events(self, events)))
    if new_game_state["step"] + 1 % 4 == 0:
        # todo update/write new model here :)
        for transition in self.transitions:
            if type(self.model) != type(np.empty(1)):
                val = np.max(self.model.predict(transition[2]))
            else:
                val = 1
            Y_t = transition[3] + LEARN_RATE * val
            update_model(self, Y_t=Y_t, old_state=transition[0], actions=transition[1], weights=None)


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
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    # todo update/write new model here :)
    for transition in self.transitions:
        if type(self.model) != type(np.empty(1)):
            val = np.max(self.model.predict(transition[2]))
            Y_t = transition[3] + LEARN_RATE * val
        elif transition[2] is None:
            Y_t = transition[3]
        else:
            val = 1
            Y_t = transition[3] + LEARN_RATE * val
        update_model(self, Y_t=Y_t, old_state=transition[0], actions=transition[1], weights=None)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.GOT_KILLED: -10,
        e.KILLED_SELF: -8,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def update_model(self, Y_t, old_state, actions, weights=None):
    '''
    :param self:
    :param Y_t:
    :param old_state: nxm Array of n game_states with dimension m each.
    :param actions:
    :param weights:
    input = np.empty((old_state.shape[0], old_state.shape[1] + 2))
    print(old_state.shape[0], old_state.shape[1] + 2, old_state.shape)
    input[:,:-2] = old_state
    input[:, -2] = np.where(ACTIONS == actions)
    input[:, -1] = Y_t
    :return:
    '''
    new_data = np.empty((1, old_state.size+2))
    print(new_data.shape, old_state.shape)
    new_data[:, :-2] = old_state
    new_data[0, -2] = ACTIONS.index(actions)
    new_data[0, -1] = Y_t
    if not os.path.isfile("my-saved-model.pt"):  #
        krr = KernelRidge(alpha=1.0)
        self.model = krr
        self.model.fit(new_data[:-1].T, new_data[-1])
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
        with open("my-saved-data.pt", "wb") as file:
            pickle.dump(new_data, file)

    else:
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        with open("my-saved-data.pt", "rb") as file:
            data_set = pickle.load(file)
        data_set = np.concatenate((data_set, new_data))
        self.model.fit(data_set[:, :-1], data_set[:, -1])
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
        with open("my-saved-data.pt", "wb") as file:
            pickle.dump(data_set, file)


''' For later perhaps?
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
# Loading some example data
X, y = load_diabetes(return_X_y=True)
# Training classifiers
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg = ereg.fit(X, y) '''