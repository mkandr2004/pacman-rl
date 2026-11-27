import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from enums import Direction
from model.Objects.Sprites.Sprite import MutantGhost
from rl.ghost_env import SlowGhostEnv, WAIT
from rl.imitation_env import make_imitation_env


class FirstLegal:
    def predict(self, observation, *, deterministic, action_masks):
        return int(np.flatnonzero(action_masks)[0]), None


class GhostControlTest(unittest.TestCase):
    def setUp(self):
        self.env = SlowGhostEnv(FirstLegal())
        self.addCleanup(self.env.close)
        observation, _ = self.env.reset(seed=95200)
        self.assertTrue(self.env.observation_space.contains(observation))
        self.game = self.env.game
        self.mapper = self.game._screen_field_mapper
        self.slow_id = self.env.slow_id
        self.ghosts = tuple(self.mapper.get_ghosts())
        self.slow = self.ghosts[self.slow_id]
        self.player = self.mapper.get_pacman()

    def test_only_commanded_ghost_leaves_its_classic_policy(self):
        x, y = self.slow.get_sprite_pos()
        self.slow._prev_step = Direction.RIGHT
        self.assertTrue(self.env.action_masks()[1])
        self.assertFalse(self.env.action_masks()[0])
        with patch.object(self.slow, "find_sprite", side_effect=AssertionError("Classic slow controller used")):
            _, _, _, _, info = self.env.step(1)
        self.assertEqual(self.slow.get_sprite_pos(), (x + 4, y))
        self.assertEqual(info["ghost_moved_frames"], {self.slow_id: 4})
        self.assertEqual(info["ghost_blocked_frames"], 0)
        self.assertTrue(all(ghost.moving_direction is not None for ghost in self.ghosts if ghost is not self.slow))

    def test_blocked_wall_and_wait_when_trapped(self):
        original = self.slow.get_sprite_pos()
        _, _, _, _, info = self.env.step(0)
        self.assertEqual(self.slow.get_sprite_pos(), original)
        self.assertEqual(info["ghost_blocked_frames"], 4)
        self.slow.move_to(0, 0)
        np.testing.assert_array_equal(self.env.action_masks(), (False, False, False, False, True))
        _, _, _, _, info = self.env.step(WAIT)
        self.assertEqual(self.slow.get_sprite_pos(), (0, 0))
        self.assertEqual(info["ghost_blocked_frames"], 0)

    def test_large_mutant_and_missing_border_still_obey_physics(self):
        mutant_id, mutant = next((i, g) for i, g in enumerate(self.ghosts) if type(g) is MutantGhost)
        self.assertEqual(mutant.get_width(), 60)
        mutant.move_to(60, 60)
        self.assertNotIn(Direction.DOWN, self.game.ghost_directions(mutant_id))
        self.game.advance(action=Direction.RIGHT, ghost_actions={mutant_id: Direction.DOWN}, render=False)
        self.assertEqual(mutant.get_sprite_pos(), (60, 60))
        self.mapper.get_walls().empty()
        x = self.mapper.get_screen_width() - mutant.get_width()
        mutant.move_to(x, 60)
        self.game.advance(action=Direction.RIGHT, ghost_actions={mutant_id: Direction.RIGHT}, render=False)
        self.assertEqual(mutant.get_sprite_pos(), (x, 60))

    def test_ghost_can_step_through_the_wall_opened_by_real_pacman_action(self):
        self.player.move_to(60, 90)
        self.slow.move_to(60, 90)
        self.player.make_breaker()
        self.player.make_invinsible()
        walls = self.mapper.get_walls()
        self.assertNotIn(Direction.DOWN, self.game.ghost_directions(self.slow_id))
        count = len(walls)
        self.game.advance(action=Direction.DOWN, ghost_actions={self.slow_id: Direction.DOWN}, render=False)
        self.assertEqual(len(walls), count - 1)
        self.assertEqual(self.slow.get_sprite_pos(), (60, 91))
        self.assertIn(Direction.DOWN, self.game.ghost_directions(self.slow_id))
        self.assertIsNone(self.game.last_capture)

    def test_contact_loses_one_life_and_invincibility_cannot_pay_again(self):
        self.slow.move_to(*self.player.get_sprite_pos())
        _, reward, ended, _, info = self.env.step(WAIT)
        self.assertFalse(ended)
        self.assertEqual(info["capture_events"], [(self.slow_id, 10)])
        self.assertEqual(info["health"], 2)
        self.assertAlmostEqual(reward, .40)
        _, reward, _, _, info = self.env.step(WAIT)
        self.assertFalse(info["capture_events"])
        self.assertEqual(info["health"], 2)
        self.assertEqual(reward, 0.)

    def test_other_ghost_capture_counts_for_team_only(self):
        other_id = next(i for i in range(len(self.ghosts)) if i != self.slow_id)
        self.ghosts[other_id].move_to(*self.player.get_sprite_pos())
        _, reward, _, _, info = self.env.step(WAIT)
        self.assertEqual(info["capture_events"], [(other_id, 10)])
        self.assertEqual(info["slow_captures"], 0)
        self.assertAlmostEqual(reward, .10)

    def test_last_life_at_time_limit_is_a_real_terminal_capture(self):
        self.env.base.max_episode_steps = 1
        self.game._controller._gamestatus.decrease_health()
        self.game._controller._gamestatus.decrease_health()
        self.slow.move_to(*self.player.get_sprite_pos())
        _, reward, ended, truncated, info = self.env.step(WAIT)
        self.assertTrue(ended)
        self.assertFalse(truncated)
        self.assertTrue(info["ghost_win"])
        self.assertAlmostEqual(reward, 1.40)

    def test_timeout_is_neutral(self):
        self.env.base.max_episode_steps = 1
        _, reward, ended, truncated, info = self.env.step(1)
        self.assertFalse(ended)
        self.assertTrue(truncated)
        self.assertTrue(info["timeout"])
        self.assertFalse(info["ghost_win"])
        self.assertEqual(reward, 0.)

    def test_classic_adapter_preserves_observations_and_frames(self):
        trace = []
        for _ in range(16):
            self.env.step_classic()
            trace.append(self.env.game.observation())
        self.env.close()
        classic = make_imitation_env(danger=False)
        self.addCleanup(classic.close)
        obs, _ = classic.reset(seed=95200)
        for expected in trace:
            action, _ = FirstLegal().predict(obs, deterministic=True, action_masks=classic.unwrapped.action_masks())
            obs, _, _, _, _ = classic.step(action)
            self.assertEqual(classic.unwrapped._game.observation(), expected)
