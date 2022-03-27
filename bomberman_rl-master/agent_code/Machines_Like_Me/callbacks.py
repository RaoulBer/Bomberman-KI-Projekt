import pickle
from collections import deque
import numpy as np


def setup(self):
    self.ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    with open("my-saved-model.pt", "rb") as file:
        self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    possible = np.array([0, 1, 2, 3, 4, 5])
    self.logger.debug("Querying model for action.")
    game_state_use = state_to_features(game_state)
    highest_known = -1000
    response = 4  # Standard: Wait
    for i in possible:
        data = np.empty((1, game_state_use.size + 1))
        data[0, :-1] = game_state_use
        data[0, -1] = i
        mod = self.model.predict(data)
        if mod > highest_known:
            highest_known = mod
            response = i
    return self.ACTIONS[response]


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        print("game state is none")
        return None

    s0, s1 = game_state["self"][3]
    features = parse_field_orientation(game_state, s0, s1)
    players = parse_players(game_state, s0, s1)
    danger = parse_danger(game_state, s0, s1)
    danger[(danger == 0).nonzero()] = features[(danger == 0).nonzero()]
    out = np.concatenate((danger, players, np.array([directionNextCoin(game_state)]).reshape(1, 1)), axis=1)
    assert out.shape[0] == 1
    return out


def parse_field_orientation(game_state: dict, s0, s1) -> np.array:
    features = np.zeros((1, 9))
    field = game_state["field"]
    mask1 = np.zeros(shape=field.shape, dtype=bool)
    mask1[(s0 - 1):(s0 + 2), (s1 - 1):(s1 + 2)] = True
    field = field[mask1.nonzero()]
    features[0, 0:field.size] = field.flatten()
    features[(features == -1).nonzero()] = 3
    assert features.shape == (1, 9)
    return features


def parse_players(game_state: dict, s0, s1) -> np.array:
    players = np.zeros((3, 2))
    others = sorted(game_state["others"], key=lambda x: (s0 - x[-1][0]) ** 2 + (s1 - x[-1][1]) ** 2)
    if others is not None:
        for i, other in enumerate(others):
            if other is None:
                break
            players[i, :] = np.array(
                [np.clip(s0 - other[-1][0], -4, 4), np.clip(s1 - other[-1][1], -4, 4)])
            # [s0 - other[-1][0], s1 - other[-1][1],  99, 99, 1])
    return players.reshape(1, 6)


def bfs(grid, start):
    goal = 99
    wall = 1
    queue = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if grid[y][x] == goal:
            return path
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= x2 < 17 and 0 <= y2 < 17 and grid[y2][x2] != wall and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))


def directionNextCoin(game_state: dict):
    s0, s1 = game_state["self"][3]
    map = game_state["field"] + game_state["explosion_map"]
    coins = sorted(game_state["coins"], key=lambda x: (s0 - x[0]) ** 2 + (s1 - x[1]) ** 2)
    if not coins:
        closest_coin = parse_crate(game_state, s0, s1)
    else:
        closest_coin = coins[0]
    if closest_coin is None:
        return -1
    map[(map != 0).nonzero()] = 1
    map[closest_coin] = 99
    path = bfs(map, (s0, s1))
    if path is None or len(path) < 2:
        out = -1
        return (out)
    else:
        assert len(path) >= 2
        next_tile = path[1]  # first element is starting point therefore the second element is the successor
    if next_tile[0] - s0 == 1:
        out = 0  # "RIGHT"
    if next_tile[0] - s0 == -1:
        out = 1  # "LEFT"
    if next_tile[1] - s1 == 1:
        out = 2  # "DOWN"
    if next_tile[1] - s1 == -1:
        out = 4  # "UP"
    return out


def parse_coins(game_state, s0, s1) -> np.array:
    coins = np.zeros((1, 2))
    c = sorted(game_state["coins"], key=lambda x: (s0 - x[0]) ** 2 + (s1 - x[1]) ** 2)
    if c:
        for i, coin in enumerate(c):
            if coin is None or i == 1:
                break
            coins[i, :] = np.array([np.clip(s0 - coin[0], -2, 2), np.clip(s1 - coin[1], -2, 2)])
    elif (game_state["field"] == 1).any():
        return np.array([parse_crate(game_state, s0, s1)[:2]]).reshape(1, 2)
    return coins.reshape(1, 2)


def parse_danger(game_state, s0, s1):
    out = np.zeros((1, 9))
    code_bomb = 2
    code_dead = 3
    b = sorted(game_state["bombs"], key=lambda x: (s0 - x[0][0]) ** 2 + (s1 - x[0][1]) ** 2)
    if b is None:
        return out
    for bomb in b:
        bombx, bomby = bomb[0]
        # 0,0
        if (s0 - 1 - bombx == 0 and s1 - 1 - bomby <= 3) or (s1 - 1 - bomby == 0 and s0 - 1 - bombx <= 3):
            out[0, 0] = code_bomb
        # 0,1
        if (s0 - 1 - bombx == 0 and s1 - bomby <= 3) or (s1 - bomby == 0 and s0 - 1 - bombx <= 3):
            out[0, 1] = code_bomb
        # 0,-1
        if (s0 - 1 - bombx == 0 and s1 + 1 - bomby <= 3) or (s1 + 1 - bomby == 0 and s0 - 1 - bombx <= 3):
            out[0, 2] = code_bomb
        # 1,0
        if (s0 - bombx == 0 and s1 - 1 - bomby == 0) or (s0 - bomb[0][0] <= 3 and s1 - 1 - bomb[0][1] <= 3):
            out[0, 3] = code_bomb
        # 1,1
        if (s0 - bombx == 0 and s1 - bomb[0][1] <= 3) or (s1 - bomby == 0 and s0 - bomb[0][0] <= 3):
            out[0, 4] = code_bomb
        # 1,-1
        if (s0 - bombx == 0 and s1 + 1 - bomby <= 3) or (s1 - bomby == 0 and s0 + 1 - bombx <= 3):
            out[0, 5] = code_bomb
        # -1,0
        if (s0 + 1 - bombx == 0 and s1 - 1 - bomby <= 3) or (s1 - 1 - bomby == 0 and s0 + 1 - 1 - bombx <= 3):
            out[0, 6] = code_bomb
        # -1,1
        if (s0 + 1 - bombx == 0 and s1 - bomby == 0) or (s0 + 1 - bomb[0][0] <= 3 and s1 - bomb[0][1] <= 3):
            out[0, 7] = code_bomb
        # -1,-1
        if (s0 + 1 - bombx == 0 and s1 + 1 - bomb[0][1] <= 3) or (s1 + 1 - bomby == 0 and s0 + 1 - bomb[0][0] <= 3):
            out[0, 8] = code_bomb

    if game_state["explosion_map"][s0 - 1, s1 - 1] != 0:
        out[0, 0] = code_dead
    if game_state["explosion_map"][s0 - 1, s1] != 0:
        out[0, 1] = code_dead
    if game_state["explosion_map"][s0 - 1, s1 + 1] != 0:
        out[0, 2] = code_dead
    if game_state["explosion_map"][s0, s1 - 1] != 0:
        out[0, 3] = code_dead
    if game_state["explosion_map"][s0, s1] != 0:
        out[0, 4] = code_dead
    if game_state["explosion_map"][s0, s1 + 1] != 0:
        out[0, 5] = code_dead
    if game_state["explosion_map"][s0 + 1, s1 - 1] != 0:
        out[0, 6] = code_dead
    if game_state["explosion_map"][s0 + 1, s1] != 0:
        out[0, 7] = code_dead
    if game_state["explosion_map"][s0 + 1, s1 + 1] != 0:
        out[0, 8] = code_dead
    return out


def parse_crate(game_state, s0, s1):
    crates = (game_state["field"] == 1).nonzero()
    c0 = crates[0]
    c1 = crates[1]
    c0a = c0 - s0
    c1a = c1 - s1
    crates = np.linalg.norm(np.stack((c0a, c1a), axis=0), axis=0)
    assert crates.size == c1a.size
    # if not crates:
    #    return np.array([0,0]).reshape(1, 2)
    if not crates.any():
        return (7, 7)
    index = np.argmin(crates)
    out = (c0[index], c1[index])
    return out
