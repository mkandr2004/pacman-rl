"""Ghost prediction using isolated copies and the engine's movement methods."""

from copy import copy
import random

import numpy as np


class ForecastCollider:
    """Same rectangle intersection as the engine, with wall rectangles cached."""

    def __init__(self, walls):
        self.rectangles = [wall.get_rect() for wall in walls]

    def check_ghost_collisions(self, ghost, obstacles):
        return ghost.get_rect().collidelist(self.rectangles) != -1


def forecast_ghosts(game, player_rectangles, seed=0):
    """Predict ghost rectangles before their move at each future collision check.

    The sampled RNG belongs to the forecast. The game RNG is never inspected.
    """
    mapper = game._screen_field_mapper
    walls = tuple(mapper.get_walls())
    controller = copy(game._controller)
    controller._collider = ForecastCollider(walls)
    ghosts = []
    for live in mapper.get_ghosts():
        ghost = copy(live)
        ghost._rect = live.get_rect().copy()
        if live._old_rect is not None:
            ghost._old_rect = live._old_rect.copy()
        ghosts.append(ghost)
    player = copy(mapper.get_pacman())
    saved_random = random.getstate()
    predictions = []
    try:
        random.seed(seed)
        for rectangle in player_rectangles:
            player._rect = rectangle
            predictions.append(tuple(ghost.get_rect().copy() for ghost in ghosts))
            controller.stupid_ghosts(ghosts, walls)
            controller.smart_ghosts(player, ghosts, walls)
    finally:
        random.setstate(saved_random)
    return predictions


def danger_map(game, rows, cols, horizon=12, samples=2):
    """Discounted tile occupancy under a stationary Pac-Man prediction."""
    grid = np.zeros((1, rows, cols), dtype=np.float32)
    if not game._screen_field_mapper.get_ghosts():
        return grid
    player = game._screen_field_mapper.get_pacman().get_rect()
    for seed in range(samples):
        for time, ghosts in enumerate(forecast_ghosts(game, [player]*horizon, seed)):
            for ghost in ghosts:
                r0, r1 = max(0, ghost.top//60), min(rows, (ghost.bottom-1)//60+1)
                c0, c1 = max(0, ghost.left//60), min(cols, (ghost.right-1)//60+1)
                grid[0, r0:r1, c0:c1] += (horizon-time)/(horizon*(horizon+1)/2*samples)
    return np.minimum(grid, 1)
