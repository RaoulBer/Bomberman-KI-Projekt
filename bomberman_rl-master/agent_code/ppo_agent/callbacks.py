import os
import pickle
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from collections import deque
import settings
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.grid import Grid
from torch.distributions.categorical import Categorical


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


#factors determining explorative behaviour
epsilon = 1.0
epsilon_decay = 0.99995
epsilon_min = 0.05


#model parameters
action_size = len(ACTIONS)
batch_size = 64
alpha = 0.0003
gamma = 0.80
n_epochs = 4
learning_rate = 0.0001
input_dims = 5
fc_input_dims = 720
fc1_dims = 360
fc2_dims = 180


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc_input_dims = fc_input_dims,
                 fc1_dims=fc1_dims, fc2_dims=fc2_dims, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.fc_input_dims = fc_input_dims
        self.n_actions = n_actions

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.conv1 = nn.Conv2d(self.input_dims, 4 * self.input_dims, 5)
        self.pool1 = nn.MaxPool2d((2, 2), ceil_mode=True)
        self.conv2 = nn.Conv2d(4 * self.input_dims, 16 * self.input_dims, 3)
        self.pool2 = nn.MaxPool2d((2, 2), ceil_mode=True)

        self.fc1 = nn.Linear(self.fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.softmax = nn.Softmax(dim=-1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.float()
        x = F.relu(self.conv1(state))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(-1, 720)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        dist = self.softmax(x)

        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc_input_dims=fc_input_dims,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.fc_input_dims = fc_input_dims
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.conv1 = nn.Conv2d(self.input_dims, 4 * self.input_dims, 5)
        self.pool1 = nn.MaxPool2d((2, 2), ceil_mode=True)
        self.conv2 = nn.Conv2d(4 * self.input_dims, 16 * self.input_dims, 3)
        self.pool2 = nn.MaxPool2d((2, 2), ceil_mode=True)

        self.fc1 = nn.Linear(self.fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.float()
        x = F.relu(self.conv1(state))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(-1, 720)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        dist = self.fc3(x)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=gamma, alpha=alpha, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=batch_size, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = observation.to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


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

    self.agent = Agent(input_dims=input_dims, n_actions=action_size, batch_size=batch_size, alpha=alpha)

    try:
        self.agent.load_models()
    except:
        return


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #self.logger.debug("Querying model for action.")
    action_chosen = ACTIONS[self.agent.choose_action(state_to_features(game_state))[0]]

    return action_chosen


def parseStep(game_state: dict) -> int:
    try:
        return T.tensor(game_state["step"])
    except ValueError:
        return 0


def parseField(game_state: dict):
    try:
        return T.tensor(game_state["field"])
    except ValueError:
        print("Value error in field parser")


def parseExplosionMap(game_state: dict):
    try:
        return T.tensor(game_state["explosion_map"])
    except ValueError:
        print("Value error in explosion map parser")


def parseBombs(game_state: dict):
    try:
        bombs = T.zeros((17, 17))
        for bomb in game_state["bombs"]:
            bombs[bomb[0][0]][bomb[0][1]] = bomb[1] + 1
        return bombs.double()
    except ValueError:
        print("Value Error in bomb parser")


def parseBombPossible(game_state: dict):
    try:
        if game_state["self"][2]:
            return 1
        else:
            return 0
    except ValueError:
        print("Value Error in bomb parser")


def parseCoins(game_state: dict):
    try:
        coinmap = T.zeros((17, 17))
        for coin in game_state["coins"]:
            coinmap[coin[0]][coin[1]] = 1
        return coinmap.double()
    except ValueError:
        print("Value Error in coin parser")


def parseSelf(game_state: dict, objective_map):
    try:
        selfmap = T.zeros((17, 17))
        if game_state["self"][2]:
            selfmap[game_state["self"][3][0]][game_state["self"][3][1]] = 1
        else:
            selfmap[game_state["self"][3][0]][game_state["self"][3][1]] = 2
        return selfmap.double()
    except ValueError:
        print("Error in Self parser")


def parseOthers(game_state: dict):
    try:
        othersmap = T.zeros((17, 17))
        for other in game_state["others"]:
            othersmap[other[3][0]][other[3][1]] = 1
        return othersmap.double()
    except ValueError:
        print("Value error in Others parser")

def state_to_features(game_state: dict):
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return T.zeros((5, 17, 17)).double()

    fieldTensor = parseField(game_state)
    coinTensor = parseCoins(game_state)
    othersTensor = parseOthers(game_state)
    bombTensor = parseBombs(game_state)
    explosionTensor = parseExplosionMap(game_state)

    concatenated = T.stack((fieldTensor,
                            coinTensor,
                            othersTensor,
                            bombTensor,
                            explosionTensor)).float()

    return concatenated
