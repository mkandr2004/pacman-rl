import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from main.lobby import CARDS, CHOICES, SIZE, choose, choose_ghosts
from main.run import create_game


class LobbyTest(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.addCleanup(pygame.quit)

    def test_keyboard_and_mouse_select_distinct_policies(self):
        surface = pygame.display.set_mode(SIZE)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1))
        self.assertEqual(CHOICES[choose(surface)].name, "DAgger 42")
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=CARDS[1].center, button=1))
        self.assertEqual(CHOICES[choose(surface)].name, "Hybride 337")
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
        self.assertIsNone(CHOICES[choose(surface)])

    def test_escape_returns_to_lobby_without_advancing_the_game(self):
        game = create_game()
        game.use_simulation_clock()
        game.enable_lobby_return()
        before = game.observation()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        after, _, info = game.advance(render=False)
        self.assertEqual(after, before)
        self.assertTrue(info["window_closed"])
        self.assertTrue(game.lobby_requested)

    def test_ghost_controller_can_be_selected_for_either_pacman(self):
        surface = pygame.display.set_mode(SIZE)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))
        self.assertEqual(choose_ghosts(surface, "DAgger 42"), 1)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1))
        self.assertEqual(choose_ghosts(surface, "Hybride 337"), 0)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.assertEqual(choose_ghosts(surface, "Clavier"), "back")

    def test_window_close_takes_priority_over_lobby_return(self):
        game = create_game()
        game.enable_lobby_return()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        _, _, info = game.advance(render=False)
        self.assertTrue(info["window_closed"])
        self.assertFalse(game.lobby_requested)
