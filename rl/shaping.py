"""Potential-based progress feedback; never executes or selects an action."""
from collections import deque
import math

import gymnasium as gym
import pygame

import consts


class ProgressPotential:
    """Distance to remaining objectives on the current dynamic maze graph."""
    COLLECTION_SCALE = 20.0
    NAVIGATION_SCALE = 2.0
    def __init__(self):
        self._cache_key = None
        self._fields = None
        self._initial_count = None

    def reset(self):
        self._cache_key = None
        self._fields = None
        self._initial_count = None

    @staticmethod
    def _field(targets, blocked, rows, cols):
        distances = {}
        queue = deque()
        for cell in targets:
            if cell not in blocked:
                distances[cell] = 0
                queue.append(cell)
        while queue:
            row, col = queue.popleft()
            for neighbor in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                r, c = neighbor
                if 0 <= r < rows and 0 <= c < cols and neighbor not in blocked and neighbor not in distances:
                    distances[neighbor] = distances[(row, col)] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, raw):
        bw, bh = consts.BLOCK_WIDTH, consts.BLOCK_HEIGHT
        width, height = raw["screen_size"]
        rows, cols = height // bh, width // bw
        count = len(raw["points"]) + len(raw["coins"])
        if self._initial_count is None:
            self._initial_count = count
        collected_fraction = 1.0 - count / max(1, self._initial_count)
        if not count:
            return self.COLLECTION_SCALE
        breaker = raw["effects"][2] > 0
        key = (raw["walls"], raw["points"], raw["coins"], raw["bonus_positions"][2], breaker)
        if key != self._cache_key:
            def cell(position):
                return int(position[1] // bh), int(position[0] // bw)
            wall_cells = set(map(cell, raw["walls"]))
            border = {(r, c) for r in range(rows) for c in range(cols)
                      if r in (0, rows-1) or c in (0, cols-1)}
            blocked = border if breaker else wall_cells | border
            targets = set(map(cell, raw["points"] + raw["coins"]))
            food = self._field(targets, blocked, rows, cols)
            lemons = self._field(set(map(cell, raw["bonus_positions"][2])), blocked, rows, cols)
            rectangles = [pygame.Rect(c*bw, r*bh, bw, bh) for r, c in blocked]
            self._fields = (food, lemons, rectangles)
            self._cache_key = key
        food, lemons, rectangles = self._fields
        x, y = raw["pacman"]
        # Connect the real pixel position to nearby graph anchors. This yields
        # feedback inside a tile too, instead of hiding 12-pixel reversals.
        def distance(field):
            candidates = []
            for r in {math.floor(y / bh), math.ceil(y / bh)}:
                for c in {math.floor(x / bw), math.ceil(x / bw)}:
                    if (r, c) not in field:
                        continue
                    ax, ay = c*bw, r*bh
                    start = pygame.Rect(x, y, consts.PACMAN_WIDTH, consts.PACMAN_HEIGHT)
                    end = pygame.Rect(ax, ay, start.width, start.height)
                    turns = (pygame.Rect(ax, y, start.width, start.height),
                             pygame.Rect(x, ay, start.width, start.height))
                    for turn in turns:
                        sweeps = (start.union(turn), turn.union(end))
                        if all(sweep.collidelist(rectangles) == -1 for sweep in sweeps):
                            candidates.append(field[(r, c)] + abs(ax-x)/bw + abs(ay-y)/bh)
                            break
            return min(candidates, default=math.inf)
        remaining_distance = distance(food)
        # A resource is a proxy only when no remaining collectible is reachable.
        # Neither the resource nor its route is turned into an action command.
        if not math.isfinite(remaining_distance):
            remaining_distance = distance(lemons)
        navigation = 0.0 if not math.isfinite(remaining_distance) else 1.0 / (1.0 + remaining_distance)
        return self.COLLECTION_SCALE * collected_fraction + self.NAVIGATION_SCALE * navigation


class PotentialShaping(gym.Wrapper):
    """Add gamma*Phi(next)-Phi(now), with zero potential on true termination."""
    def __init__(self, env, gamma=0.995):
        super().__init__(env)
        self.gamma = gamma
        self.potential = ProgressPotential()
        self._previous = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.potential.reset()
        self._previous = self.potential(self.env.unwrapped._last_raw_observation)
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        next_potential = 0.0 if terminated else self.potential(self.env.unwrapped._last_raw_observation)
        shaping = self.gamma * next_potential - self._previous
        info = dict(info)
        info["base_reward"] = base_reward
        info["potential_before"] = self._previous
        info["potential_after"] = next_potential
        info["reward_terms"] = {**info["reward_terms"], "potential": shaping}
        self._previous = next_potential
        return obs, float(base_reward + shaping), terminated, truncated, info
