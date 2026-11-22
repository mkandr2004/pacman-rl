"""Run a tiny deterministic scenario without Pygame or reinforcement learning."""

from pac_ai_arena.core import Direction, GameConfig, GameEngine, GameState


def describe(state: GameState) -> str:
    return (
        f"step={state.step_count} "
        f"pacman=({state.pacman.row},{state.pacman.column}) "
        f"ghost=({state.ghost.row},{state.ghost.column}) "
        f"pellets={len(state.pellets)} "
        f"score={state.score} "
        f"outcome={state.outcome.value}"
    )


def main() -> None:
    config = GameConfig.from_ascii(
        (
            "#######",
            "#P...G#",
            "#.....#",
            "#######",
        ),
        max_steps=10,
    )
    engine = GameEngine(config)
    state = engine.reset(seed=42)
    print(f"initial: {describe(state)}")

    actions = (
        (Direction.RIGHT, Direction.LEFT),
        (Direction.RIGHT, Direction.LEFT),
    )
    for pacman_action, ghost_action in actions:
        result = engine.step(state, pacman_action, ghost_action)
        event_names = ", ".join(event.value for event in result.events) or "none"
        print(f"events: {event_names}")
        print(f"state:   {describe(result.state)}")
        state = result.state


if __name__ == "__main__":
    main()
