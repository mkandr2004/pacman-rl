"""Immutable data types for the deterministic game core."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Direction(Enum):
    """The four actions available to each actor."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Outcome(Enum):
    """Mutually exclusive episode outcomes."""

    IN_PROGRESS = "in_progress"
    PACMAN_WIN = "pacman_win"
    GHOST_WIN = "ghost_win"
    TIMEOUT = "timeout"


class Event(Enum):
    """Factual events emitted by one game tick."""

    PACMAN_BLOCKED = "pacman_blocked"
    GHOST_BLOCKED = "ghost_blocked"
    PELLET_EATEN = "pellet_eaten"
    CAPTURE = "capture"
    ALL_PELLETS_EATEN = "all_pellets_eaten"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True, order=True)
class Position:
    """A row and column in the grid."""

    row: int
    column: int

    def moved(self, direction: Direction) -> Position:
        row_delta, column_delta = direction.value
        return Position(self.row + row_delta, self.column + column_delta)


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Fixed map and episode limit used by the game engine."""

    height: int
    width: int
    walls: frozenset[Position]
    pacman_start: Position
    ghost_start: Position
    initial_pellets: frozenset[Position]
    max_steps: int

    @classmethod
    def from_ascii(cls, rows: Iterable[str], *, max_steps: int) -> GameConfig:
        """Build a config from ``#`` walls, ``P``, ``G``, ``.`` and spaces."""

        parsed_rows = tuple(rows)
        if not parsed_rows or not parsed_rows[0]:
            raise ValueError("map must not be empty")
        if any(len(row) != len(parsed_rows[0]) for row in parsed_rows):
            raise ValueError("map must be rectangular")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        allowed = {"#", "P", "G", ".", " "}
        walls: set[Position] = set()
        pellets: set[Position] = set()
        pacman_positions: list[Position] = []
        ghost_positions: list[Position] = []

        for row_index, row in enumerate(parsed_rows):
            for column_index, cell in enumerate(row):
                position = Position(row_index, column_index)
                if cell not in allowed:
                    raise ValueError(f"unsupported map character: {cell!r}")
                if cell == "#":
                    walls.add(position)
                elif cell == ".":
                    pellets.add(position)
                elif cell == "P":
                    pacman_positions.append(position)
                elif cell == "G":
                    ghost_positions.append(position)

        if len(pacman_positions) != 1:
            raise ValueError("map must contain exactly one Pac-Man")
        if len(ghost_positions) != 1:
            raise ValueError("map must contain exactly one ghost")
        if not pellets:
            raise ValueError("map must contain at least one pellet")

        return cls(
            height=len(parsed_rows),
            width=len(parsed_rows[0]),
            walls=frozenset(walls),
            pacman_start=pacman_positions[0],
            ghost_start=ghost_positions[0],
            initial_pellets=frozenset(pellets),
            max_steps=max_steps,
        )

    def is_walkable(self, position: Position) -> bool:
        return (
            0 <= position.row < self.height
            and 0 <= position.column < self.width
            and position not in self.walls
        )


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete state required to continue one episode."""

    pacman: Position
    ghost: Position
    pellets: frozenset[Position]
    step_count: int
    score: int
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class StepResult:
    """New state plus factual events, without any RL reward."""

    state: GameState
    events: tuple[Event, ...]
