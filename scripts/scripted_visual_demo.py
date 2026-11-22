"""Demonstrate external control in the original visual Pygame game."""

import argparse
from itertools import cycle

from enums import Direction
from main.run import main as run_game


class ScriptedPolicy:
    """Cycle through directions; this is a control test, not an AI agent."""

    def __init__(self, frames_per_direction=45):
        self._directions = cycle(
            (Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP)
        )
        self._frames_per_direction = frames_per_direction
        self._frame = 0
        self._direction = next(self._directions)

    def __call__(self, observation):
        _ = observation
        if self._frame and self._frame % self._frames_per_direction == 0:
            self._direction = next(self._directions)
        self._frame += 1
        return self._direction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop automatically after this many rendered frames.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    final_observation = run_game(
        action_provider=ScriptedPolicy(),
        max_frames=args.max_frames,
    )
    print(final_observation)
