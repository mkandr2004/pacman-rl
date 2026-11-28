import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from enums import Direction
from main.run import create_game


class GhostContactTest(unittest.TestCase):
    def test_ghost_entering_pacman_hitbox_costs_one_life_on_that_frame(self):
        game = create_game(action_provider=lambda observation: None)
        self.addCleanup(pygame.quit)
        game.use_simulation_clock()
        mapper = game._screen_field_mapper
        pacman = mapper.get_pacman()
        ghosts = tuple(mapper.get_ghosts())
        ghost = ghosts[0]
        mapper.get_walls().empty()
        for other in ghosts:
            other.move_to(300, 300)
        pacman.move_to(60, 60)
        ghost.move_to(60 + pacman.get_rect().width, 60)
        self.assertFalse(pacman.collide(ghost))
        health = game.observation()["health"]
        actions = dict.fromkeys(range(len(ghosts)))
        actions[0] = Direction.LEFT

        observation, _, _ = game.advance(render=False, ghost_actions=actions)

        self.assertTrue(pacman.collide(ghost))
        self.assertEqual(observation["health"], health - 1)
        self.assertIs(game.last_capture, ghost)

        observation, _, _ = game.advance(render=False, ghost_actions=actions)

        self.assertEqual(observation["health"], health - 1)
        self.assertIsNone(game.last_capture)
