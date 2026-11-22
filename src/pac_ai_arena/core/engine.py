"""Pure, deterministic transition rules for PAC-AI ARENA."""

from __future__ import annotations

from pac_ai_arena.core.model import (
    Direction,
    Event,
    GameConfig,
    GameState,
    Outcome,
    Position,
    StepResult,
)


class GameEngine:
    """Apply simultaneous actions to immutable game states."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config

    def reset(self, *, seed: int | None = None) -> GameState:
        """Return the initial state.

        The phase-1 map is fixed, so ``seed`` does not alter the state yet. The
        parameter is part of the contract for future seeded randomisation.
        """

        _ = seed
        return GameState(
            pacman=self.config.pacman_start,
            ghost=self.config.ghost_start,
            pellets=self.config.initial_pellets,
            step_count=0,
            score=0,
            outcome=Outcome.IN_PROGRESS,
        )

    def step(
        self,
        state: GameState,
        pacman_action: Direction,
        ghost_action: Direction,
    ) -> StepResult:
        """Apply both actions on one tick and resolve its events."""

        if state.outcome is not Outcome.IN_PROGRESS:
            raise ValueError("a terminal state cannot be advanced")
        if not isinstance(pacman_action, Direction) or not isinstance(ghost_action, Direction):
            raise TypeError("actions must be Direction values")

        events: list[Event] = []
        next_pacman = self._destination(state.pacman, pacman_action)
        next_ghost = self._destination(state.ghost, ghost_action)

        if next_pacman == state.pacman:
            events.append(Event.PACMAN_BLOCKED)
        if next_ghost == state.ghost:
            events.append(Event.GHOST_BLOCKED)

        pellets = state.pellets
        score = state.score
        if next_pacman in pellets:
            pellets = pellets - {next_pacman}
            score += 1
            events.append(Event.PELLET_EATEN)

        same_destination = next_pacman == next_ghost
        crossed_positions = next_pacman == state.ghost and next_ghost == state.pacman
        captured = same_destination or crossed_positions
        if captured:
            events.append(Event.CAPTURE)

        all_pellets_eaten = not pellets
        if all_pellets_eaten:
            events.append(Event.ALL_PELLETS_EATEN)

        step_count = state.step_count + 1
        if captured:
            outcome = Outcome.GHOST_WIN
        elif all_pellets_eaten:
            outcome = Outcome.PACMAN_WIN
        elif step_count >= self.config.max_steps:
            outcome = Outcome.TIMEOUT
            events.append(Event.TIMEOUT)
        else:
            outcome = Outcome.IN_PROGRESS

        return StepResult(
            state=GameState(
                pacman=next_pacman,
                ghost=next_ghost,
                pellets=pellets,
                step_count=step_count,
                score=score,
                outcome=outcome,
            ),
            events=tuple(events),
        )

    def _destination(self, position: Position, direction: Direction) -> Position:
        candidate = position.moved(direction)
        return candidate if self.config.is_walkable(candidate) else position
