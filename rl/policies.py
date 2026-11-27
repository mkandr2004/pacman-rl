"""Understandable non-learning policies used as RL baselines."""

from collections import deque

import numpy as np

import consts
from rl.pacman_env import PacmanEnv


class HeuristicPolicy:
    """Follow shortest safe paths to collectibles on the symbolic grid."""

    ACTION_DELTAS = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
    }

    def __init__(self, danger_radius=1):
        if danger_radius < 0:
            raise ValueError("danger_radius cannot be negative")
        self.danger_radius = danger_radius
        self._current_action = None

    def reset(self):
        self._current_action = None

    @staticmethod
    def _cell_from_pixels(position):
        x, y = (int(value) for value in position)
        return y // consts.BLOCK_HEIGHT, x // consts.BLOCK_WIDTH

    @staticmethod
    def _inside(cell, rows, cols):
        row, col = cell
        return 0 <= row < rows and 0 <= col < cols

    def _danger_cells(self, ghost_positions, rows, cols):
        danger = set()
        for position in ghost_positions:
            ghost_row, ghost_col = self._cell_from_pixels(position)
            for row in range(rows):
                for col in range(cols):
                    if abs(row - ghost_row) + abs(col - ghost_col) <= self.danger_radius:
                        danger.add((row, col))
        return danger

    def _first_path_action(self, start, targets, blocked, rows, cols):
        queue = deque([(start, None)])
        visited = {start}

        while queue:
            cell, first_action = queue.popleft()
            if cell in targets and cell != start:
                return first_action

            for action, (delta_row, delta_col) in self.ACTION_DELTAS.items():
                neighbor = (cell[0] + delta_row, cell[1] + delta_col)
                if (
                    self._inside(neighbor, rows, cols)
                    and neighbor not in blocked
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    queue.append(
                        (neighbor, action if first_action is None else first_action)
                    )
        return None

    def _safest_legal_action(self, start, walls, ghost_cells, rows, cols):
        candidates = []
        for action, (delta_row, delta_col) in self.ACTION_DELTAS.items():
            neighbor = (start[0] + delta_row, start[1] + delta_col)
            if not self._inside(neighbor, rows, cols) or neighbor in walls:
                continue
            nearest_ghost = min(
                (
                    abs(neighbor[0] - ghost[0]) + abs(neighbor[1] - ghost[1])
                    for ghost in ghost_cells
                ),
                default=rows + cols,
            )
            candidates.append((nearest_ghost, -action, action))

        if not candidates:
            return 0
        return max(candidates)[2]

    def predict(self, observation):
        pacman_x, pacman_y = (
            int(value) for value in observation["pacman_position"]
        )
        aligned = (
            pacman_x % consts.BLOCK_WIDTH == 0
            and pacman_y % consts.BLOCK_HEIGHT == 0
        )
        if not aligned and self._current_action is not None:
            return self._current_action

        grid = observation["grid"]
        _, rows, cols = grid.shape
        start = self._cell_from_pixels(observation["pacman_position"])
        walls = set(map(tuple, np.argwhere(grid[PacmanEnv.WALLS] == 1)))
        targets = set(
            map(
                tuple,
                np.argwhere(
                    (grid[PacmanEnv.COINS] == 1)
                    | (grid[PacmanEnv.POINTS] == 1)
                ),
            )
        )
        ghost_positions = observation["ghost_positions"].reshape(-1, 2)
        ghost_cells = {
            self._cell_from_pixels(position)
            for position in ghost_positions
        }
        danger = self._danger_cells(
            ghost_positions, rows, cols
        )

        safe_blocked = walls | danger
        safe_blocked.discard(start)
        action = self._first_path_action(
            start, targets, safe_blocked, rows, cols
        )
        if action is None:
            action = self._first_path_action(start, targets, walls, rows, cols)
        if action is None:
            action = self._safest_legal_action(
                start, walls, ghost_cells, rows, cols
            )

        self._current_action = action
        return action
