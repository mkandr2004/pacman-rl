"""Grid pursuit and interception policy for the slow and fast ghost pair."""

from collections import deque

import numpy as np

import consts
from rl.ghost_env import DIRECTIONS, WAIT


DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
SEPARATION_WEIGHT = 1.5
GHOST_CLEARANCE = 6


def neighbors(cell, walls):
    for dy, dx in DELTAS:
        y, x = cell[0] + dy, cell[1] + dx
        if 0 <= y < walls.shape[0] and 0 <= x < walls.shape[1] and not walls[y, x]:
            yield y, x


def interception_actions(env, masks):
    """Send the slow ghost after Pac-Man and the fast one to the next junction."""
    raw = env.base._last_raw_observation
    walls = env.match._pacman_observation["grid"][0]
    target = env.base._grid_position(raw["pacman"])
    mapper = env.match.game._screen_field_mapper
    heading = mapper.get_pacman().moving_direction
    ahead = target
    if heading in DIRECTIONS:
        dy, dx = DELTAS[DIRECTIONS.index(heading)]
        for offset in range(1, 5):
            candidate = target[0] + dy * offset, target[1] + dx * offset
            if candidate not in neighbors(ahead, walls):
                break
            ahead = candidate
            if sum(1 for _ in neighbors(ahead, walls)) >= 3:
                break

    actions = []
    ghosts = list(mapper.get_ghosts())
    positions = [ghost.get_rect().copy() for ghost in ghosts]
    for slot, destination, mask in zip(env.slots, (target, ahead), masks, strict=True):
        if mask[WAIT]:
            actions.append(WAIT)
            continue
        distance = {destination: 0}
        queue = deque([destination])
        while queue:
            cell = queue.popleft()
            for neighbor in neighbors(cell, walls):
                if neighbor not in distance:
                    distance[neighbor] = distance[cell] + 1
                    queue.append(neighbor)
        ghost = ghosts[slot]
        x, y = raw["ghosts"][slot]
        width, height = consts.BLOCK_WIDTH, consts.BLOCK_HEIGHT
        origin = round(y / height), round(x / width)
        if origin != destination:
            destination = min(neighbors(origin, walls),
                              key=lambda cell: distance.get(cell, walls.size), default=origin)

        def offset(action):
            dy, dx = DELTAS[action]
            return dx * ghost.speed_horizontal, dy * ghost.speed_vertical

        def rank(action):
            dx, dy = offset(action)
            route = (abs(x + dx - destination[1] * width)
                     + abs(y + dy - destination[0] * height))
            candidate = positions[slot].move(dx, dy)
            nearby = 0
            for other, rect in enumerate(positions):
                if other == slot:
                    continue
                gap = max(rect.left - candidate.right, candidate.left - rect.right,
                          rect.top - candidate.bottom, candidate.top - rect.bottom)
                nearby = max(nearby, GHOST_CLEARANCE - gap)
            return route + SEPARATION_WEIGHT * nearby, DIRECTIONS[action] != ghost.moving_direction, action

        action = min(np.flatnonzero(mask), key=rank)
        actions.append(action)
        dx, dy = offset(action)
        positions[slot].move_ip(dx, dy)
    return np.asarray(actions)
