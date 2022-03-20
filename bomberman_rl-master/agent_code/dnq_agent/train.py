from collections import namedtuple, deque

import pickle
import numpy as np
import random
from typing import List
import os

from torchinfo import summary

import events as e
import settings
from . import callbacks
import torch.optim as optim

import torch as T


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 300  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
batch_size = 64

# Events
PLACEHOLDER = "PLACEHOLDER"

#Training parameters
epochs_per_state = 1
training_verbosity = 0

ROUNDS = 0


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')

    self.states = np.zeros((TRANSITION_HISTORY_SIZE, 242), dtype=np.float32)
    self.nextstates = np.zeros((TRANSITION_HISTORY_SIZE, 242), dtype=np.float32)

    self.actions = np.zeros(TRANSITION_HISTORY_SIZE, dtype=np.int32)
    self.rewards = np.zeros(TRANSITION_HISTORY_SIZE, dtype=np.int32)
    self.terminals = np.zeros(TRANSITION_HISTORY_SIZE, dtype=np.int32)

    self.MEMORY_ITERATOR = 0
    self.ITERATION_COUNTER = 0
    self.gamma = callbacks.gamma
    self.batch_size = batch_size
    self.mem_size = TRANSITION_HISTORY_SIZE

    #self.scheduler = T.optim.lr_scheduler.MultiplicativeLR(self.model.optimizer, lambda x: 0.99995, verbose=True)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    #remember
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
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #Leaving this out for now
    #if False:
     #   events.append("WAITED")

    # state_to_features is defined in callbacks.py


    idx = (self.MEMORY_ITERATOR % TRANSITION_HISTORY_SIZE) - 1

    if old_game_state == None:
        pass

    else:
        self.states[idx] = callbacks.state_to_features(old_game_state)
        self.nextstates[idx] = callbacks.state_to_features(new_game_state)
        self.rewards[idx] = reward_from_events(self, events)
        self.actions[idx] = callbacks.ACTIONS.index(self_action)
        self.terminals[idx] = 0

        self.MEMORY_ITERATOR += 1

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
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    idx = (self.MEMORY_ITERATOR % TRANSITION_HISTORY_SIZE) - 1
    self.states[idx] = callbacks.state_to_features(last_game_state)
    self.nextstates[idx] = callbacks.state_to_features(None)
    self.rewards[idx] = reward_from_events(self, events)
    self.actions[idx] = callbacks.ACTIONS.index(last_action)
    self.terminals[idx] = 1

    self.MEMORY_ITERATOR += 1

    if self.MEMORY_ITERATOR < self.batch_size:
        pass

    else:
        self.model.optimizer.zero_grad()
        self.model.train(mode=True)     #For Batch normalization and pooling layers

        max_mem = min(self.MEMORY_ITERATOR, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_idx = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.states[batch]).to(self.model.device)
        new_state_batch = T.tensor(self.nextstates[batch]).to(self.model.device)
        action_batch = self.actions[batch]
        reward_batch = T.tensor(self.rewards[batch]).to(self.model.device)
        terminal_batch = T.tensor(self.terminals[batch]).to(self.model.device)

        evaled_temp = self.model.forward(state_batch)
        evaled = evaled_temp[batch_idx, action_batch]
        next = self.model.forward(new_state_batch)
        next[terminal_batch.long()] = 0.0

        target = reward_batch + self.gamma * T.max(next, dim=1)[0]

        print("Reward: ", T.sum(reward_batch), end="")

        loss = self.model.loss(target, evaled).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()
        T.autograd.set_detect_anomaly(True)

        self.ITERATION_COUNTER += 1
        print("epsilon: ", callbacks.epsilon)
        callbacks.epsilon = callbacks.epsilon * callbacks.epsilon_decay if callbacks.epsilon > callbacks.epsilon_min \
            else callbacks.epsilon_min


        if self.ITERATION_COUNTER % 100 == 0:
            T.save(self.model, "model.pt")
            print(summary(self.model,[64,242],depth=4))


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.CRATE_DESTROYED: 5,
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 10,
        e.GOT_KILLED: -10,
        e.KILLED_SELF: -12,
        e.WAITED: -5,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: -2,
        e.SURVIVED_ROUND: 20,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
