import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from enums import Direction
from main.run import main as run_game


class ExternalControlTest(unittest.TestCase):
    def test_keyboard_mode_still_starts_without_an_external_policy(self):
        observation = run_game(max_frames=5)

        self.assertEqual(observation["pacman"], (60, 60))
        self.assertEqual(observation["score"], 0)

    def test_external_policy_moves_pacman_in_the_visual_engine(self):
        observation = run_game(
            action_provider=lambda _: Direction.RIGHT,
            max_frames=10,
        )

        self.assertEqual(observation["pacman"], (90, 60))
        self.assertEqual(observation["coins_remaining"], 11)
        self.assertEqual(observation["points_remaining"], 81)


if __name__ == "__main__":
    unittest.main()
