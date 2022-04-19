from collections import namedtuple, deque

import pickle
import numpy as np
import random
from typing import List
import os

from torchinfo import summary

import events as e
import settings
from . import callbacks as c
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torch as T


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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
    self.ITERATION_COUNTER = 0
    self.rewardlist = []
    self.tensorboardwriter = SummaryWriter()


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

    if old_game_state == None:
        return

    else:
        observation = c.state_to_features(old_game_state)
        action, probs, value = self.agent.choose_action(observation)
        reward = reward_from_events(self, events)
        self.rewardlist.append(reward)
        c.Agent.remember(self.agent, observation, action, probs, value, reward, 0)
        self.ITERATION_COUNTER += 1


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

    ##log lenght of rounds
    self.tensorboardwriter.add_scalar("Roundlength in Steps", last_game_state["step"], self.ITERATION_COUNTER)

    observation = c.state_to_features(last_game_state)
    action, probs, value = self.agent.choose_action(observation)
    reward = reward_from_events(self, events)
    self.rewardlist.append(reward)
    self.tensorboardwriter.add_scalar("Episode total reward", sum(self.rewardlist), self.ITERATION_COUNTER)
    self.rewardlist = []
    c.Agent.remember(self.agent, observation, action, probs, value, reward, 1)

    if self.ITERATION_COUNTER % c.batch_size == 0:
        self.agent.learn()
        self.agent.save_models()
    self.ITERATION_COUNTER += 1


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.CRATE_DESTROYED: 10,
        e.COIN_COLLECTED: 15,
        e.KILLED_OPPONENT: 30,
        e.GOT_KILLED: -10,
        e.KILLED_SELF: -30,
        e.WAITED: -5,
        e.INVALID_ACTION: -2,
        e.BOMB_DROPPED: -1,
        e.SURVIVED_ROUND: 1,
        e.IN_DANGER: -2
    }
    reward_sum = 0
    for event in events:
        #self.tensorboardwriter.add_histogram("events per step", event, self.ITERATION_COUNTER)
        if event in game_rewards:
            reward_sum += game_rewards[event]

    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
