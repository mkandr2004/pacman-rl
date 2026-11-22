import pytest

from pac_ai_arena.core import (
    Direction,
    Event,
    GameConfig,
    GameEngine,
    Outcome,
    Position,
)


def engine_from(*rows: str, max_steps: int = 20) -> GameEngine:
    return GameEngine(GameConfig.from_ascii(rows, max_steps=max_steps))


def test_reset_is_deterministic_for_a_fixed_map() -> None:
    engine = engine_from(
        "#####",
        "#P.G#",
        "#...#",
        "#####",
    )

    assert engine.reset(seed=42) == engine.reset(seed=42)


def test_walls_keep_both_actors_in_place() -> None:
    engine = engine_from(
        "#####",
        "#P.G#",
        "#...#",
        "#####",
    )
    state = engine.reset(seed=42)

    result = engine.step(state, Direction.LEFT, Direction.RIGHT)

    assert result.state.pacman == Position(1, 1)
    assert result.state.ghost == Position(1, 3)
    assert result.events == (Event.PACMAN_BLOCKED, Event.GHOST_BLOCKED)


def test_two_actions_are_applied_on_the_same_tick() -> None:
    engine = engine_from(
        "#####",
        "#P.G#",
        "#...#",
        "#####",
    )
    state = engine.reset(seed=42)

    result = engine.step(state, Direction.DOWN, Direction.LEFT)

    assert result.state.pacman == Position(2, 1)
    assert result.state.ghost == Position(1, 2)
    assert result.state.step_count == 1
    assert result.state.outcome is Outcome.IN_PROGRESS


def test_same_destination_is_a_capture() -> None:
    engine = engine_from(
        "#####",
        "#P.G#",
        "#...#",
        "#####",
    )
    state = engine.reset(seed=42)

    result = engine.step(state, Direction.RIGHT, Direction.LEFT)

    assert result.state.pacman == result.state.ghost == Position(1, 2)
    assert Event.CAPTURE in result.events
    assert result.state.outcome is Outcome.GHOST_WIN


def test_crossing_positions_is_a_capture() -> None:
    engine = engine_from(
        "#####",
        "#PG.#",
        "#####",
    )
    state = engine.reset(seed=42)

    result = engine.step(state, Direction.RIGHT, Direction.LEFT)

    assert result.state.pacman == Position(1, 2)
    assert result.state.ghost == Position(1, 1)
    assert Event.CAPTURE in result.events
    assert result.state.outcome is Outcome.GHOST_WIN


def test_eating_the_last_pellet_wins_the_game() -> None:
    engine = engine_from(
        "#####",
        "#P.G#",
        "#####",
    )
    state = engine.reset(seed=42)

    result = engine.step(state, Direction.RIGHT, Direction.RIGHT)

    assert result.state.score == 1
    assert result.state.pellets == frozenset()
    assert result.events == (Event.GHOST_BLOCKED, Event.PELLET_EATEN, Event.ALL_PELLETS_EATEN)
    assert result.state.outcome is Outcome.PACMAN_WIN


def test_capture_has_priority_when_the_last_pellet_is_eaten() -> None:
    engine = engine_from(
        "#####",
        "#P.G#",
        "#####",
    )
    state = engine.reset(seed=42)

    result = engine.step(state, Direction.RIGHT, Direction.LEFT)

    assert result.state.score == 1
    assert result.state.pellets == frozenset()
    assert Event.PELLET_EATEN in result.events
    assert Event.ALL_PELLETS_EATEN in result.events
    assert Event.CAPTURE in result.events
    assert result.state.outcome is Outcome.GHOST_WIN


def test_step_limit_is_a_draw_not_a_win() -> None:
    engine = engine_from(
        "#####",
        "#P G#",
        "# . #",
        "#####",
        max_steps=1,
    )
    state = engine.reset(seed=42)

    result = engine.step(state, Direction.DOWN, Direction.UP)

    assert result.state.step_count == 1
    assert result.state.outcome is Outcome.TIMEOUT
    assert Event.TIMEOUT in result.events
    assert Event.CAPTURE not in result.events


def test_a_terminal_state_cannot_be_advanced() -> None:
    engine = engine_from(
        "#####",
        "#P.G#",
        "#####",
    )
    state = engine.reset(seed=42)
    terminal = engine.step(state, Direction.RIGHT, Direction.RIGHT).state

    with pytest.raises(ValueError, match="terminal"):
        engine.step(terminal, Direction.LEFT, Direction.LEFT)


@pytest.mark.parametrize(
    "rows, message",
    [
        (("####", "#PG#", "###"), "rectangular"),
        (("#####", "#PPG#", "#...#", "#####"), "exactly one Pac-Man"),
        (("#####", "#P.G#", "#?###"), "unsupported"),
    ],
)
def test_invalid_maps_are_rejected(rows: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GameConfig.from_ascii(rows, max_steps=20)
