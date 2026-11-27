import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from enums import Direction
from rl.pacman_env import PacmanEnv


class BorderTest(unittest.TestCase):
    def setUp(self):
        self.env = PacmanEnv(max_episode_steps=1000)
        self.env.reset(seed=4)
        self.mapper = self.env._game._screen_field_mapper
        self.player = self.mapper.get_pacman()
        self.controller = self.env._game._controller

    def tearDown(self):
        self.env.close()

    def move(self, direction):
        return self.controller.player_action(self.player, self.mapper.get_walls(),
                                             self.mapper.get_container(), direction)

    def test_all_four_borders_are_indestructible_with_lemon(self):
        original = tuple(self.mapper.get_walls())
        self.player.make_breaker()
        for x, y, direction in ((60, 60, Direction.LEFT), (60, 60, Direction.UP),
                                (540, 630, Direction.DOWN), (870, 60, Direction.RIGHT)):
            self.player.move_to(x, y)
            self.assertTrue(self.move(direction))
            self.assertEqual(self.player.get_sprite_pos(), (x, y))
            self.assertEqual(tuple(self.mapper.get_walls()), original)

    def test_bottom_exit_is_blocked_even_if_border_is_missing(self):
        self.mapper.get_walls().empty()
        self.player.move_to(540, 690)
        self.player.make_breaker()
        for _ in range(12):
            self.move(Direction.DOWN)
        self.assertEqual(self.player.get_sprite_pos(), (540, 690))

    def test_interior_wall_requires_active_lemon(self):
        self.player.move_to(60, 90)
        count = len(self.mapper.get_walls())
        self.assertTrue(self.move(Direction.DOWN))
        self.assertEqual(len(self.mapper.get_walls()), count)
        self.player.make_breaker()
        self.assertTrue(self.move(Direction.DOWN))
        self.assertEqual(len(self.mapper.get_walls()), count - 1)
        self.player.check_breaker(3000)
        self.assertFalse(self.player.is_breaker())
        # Another interior wall at (180,120) must no longer be breakable.
        self.player.move_to(180, 90)
        count = len(self.mapper.get_walls())
        self.assertTrue(self.move(Direction.DOWN))
        self.assertEqual(len(self.mapper.get_walls()), count)

    def test_action_repeat_cannot_cross_bottom_and_timeout_is_not_victory(self):
        self.env.max_episode_steps = 1
        self.player.move_to(540, 630)
        self.player.make_breaker()
        _, _, terminated, truncated, info = self.env.step(3)
        self.assertEqual(self.player.get_sprite_pos(), (540, 630))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["victory"])

    def test_keyboard_control_uses_the_same_border_protection(self):
        self.player.move_to(540, 630)
        self.player.make_breaker()
        self.controller._down = True
        with patch.object(self.controller, "_update_buttons"):
            self.assertTrue(self.controller.player_game(self.player, self.mapper.get_walls()))
        self.assertEqual(self.player.get_sprite_pos(), (540, 630))

    def test_collection_is_rewarded_once_and_display_score_is_unchanged(self):
        self.env.step(1)
        self.env.step(1)
        _, reward, _, _, info = self.env.step(1)
        self.assertEqual(info["score"], 1)
        self.assertEqual(info["reward_terms"]["score"], 3)
        _, reward, _, _, info = self.env.step(1)
        self.assertEqual(info["score"], 1)
        self.assertEqual(info["reward_terms"]["score"], 0)

    def test_victory_requires_points_and_coins_but_not_fruit(self):
        field = self.mapper.get_field()
        field._points_found = field._total_points
        self.assertFalse(field.is_victory())
        field._coins_found = field._total_coins
        self.assertTrue(field.is_victory())
        field._points_found -= 1
        self.assertFalse(field.is_victory())
