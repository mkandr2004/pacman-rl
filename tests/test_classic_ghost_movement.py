import os
import random
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

import consts
from main.run import create_game
from model.Objects.Sprites.Sprite import SleepingGhost


class ClassicGhostMovementTest(unittest.TestCase):
    def test_sleeping_ghosts_leave_the_corner_with_pacman_stationary(self):
        self.addCleanup(random.setstate, random.getstate())
        random.seed(92000)
        game = create_game()
        self.addCleanup(pygame.quit)
        game.use_simulation_clock()
        ghosts = [ghost for ghost in game._screen_field_mapper.get_ghosts()
                  if isinstance(ghost, SleepingGhost)]
        self.assertEqual(len(ghosts), 2)
        traces = [[] for ghost in ghosts]

        for frame in range(600):
            game.advance(render=False, process_events=False)
            if frame >= 500:
                for ghost, trace in zip(ghosts, traces):
                    trace.append(ghost.get_sprite_pos())

        for trace in traces:
            with self.subTest(start=trace[0]):
                xs, ys = zip(*trace)
                self.assertGreaterEqual(max(xs) - min(xs) + max(ys) - min(ys),
                                        consts.BLOCK_WIDTH // 2)
