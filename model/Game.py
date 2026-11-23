import pygame

import consts
from enums import Mode
from model.Events.MoveEvent import PacmanMoveEvent, CoinTossEvent, GhostAnimEvent


class Game:
    def __init__(self, mode, action_provider=None):
        self._mode = mode
        self._action_provider = action_provider
        self._clock = pygame.time.Clock()

    def set_screen_field_mapper(self, screen_field_mapper):
        self._screen_field_mapper = screen_field_mapper

    def set_drawer(self, drawer):
        self._drawer = drawer

    def set_controller(self, controller):
        self._controller = controller

    def _process_events(self):
        active = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                active = False
            if event.type == PacmanMoveEvent.get_event_id():
                self._screen_field_mapper.get_pacman().next_state()
            elif event.type == CoinTossEvent.get_event_id():
                self._screen_field_mapper.change_coins_state()
            elif event.type == GhostAnimEvent.get_event_id():
                self._screen_field_mapper.change_ghosts_state()
        return active

    def _check_invincible(self):
        pacman = self._screen_field_mapper.get_pacman()
        if pacman.is_invinsible():
            end_time = pygame.time.get_ticks()
            pacman.check_invinsible(end_time)

    def _check_speed(self):
        pacman = self._screen_field_mapper.get_pacman()
        if pacman.is_speeded():
            end_time = pygame.time.get_ticks()
            pacman.check_speed(end_time)

    def _check_breaker(self):
        pacman = self._screen_field_mapper.get_pacman()
        if pacman.is_breaker():
            end_time = pygame.time.get_ticks()
            pacman.check_breaker(end_time)

    def _check_statuses(self):
        self._check_invincible()
        self._check_speed()
        self._check_breaker()

    def _check_gameover(self):
        if self._screen_field_mapper.is_victory():
            self._drawer.draw_victory()
            return True
        elif self._controller.is_gameover():
            self._drawer.draw_gameover(self._controller.score())
            return True
        return False

    def _player_game(self, action=None):
        pacman = self._screen_field_mapper.get_pacman()
        walls = self._screen_field_mapper.get_walls()
        container = self._screen_field_mapper.get_container()

        if action is not None:
            self._controller.player_action(pacman, walls, container, action)
        elif self._action_provider is None:
            if not self._controller.player_game(pacman, walls):
                self._controller.player_game_group(pacman, container)
        else:
            provided_action = self._action_provider(self.observation())
            if provided_action is not None:
                self._controller.player_action(pacman, walls, container, provided_action)

        self._controller.ghosts_player(pacman, self._screen_field_mapper.get_ghosts())
        self._controller.stupid_ghosts(self._screen_field_mapper.get_ghosts(), self._screen_field_mapper.get_walls())
        self._controller.smart_ghosts(pacman,
                                      self._screen_field_mapper.get_ghosts(),
                                      self._screen_field_mapper.get_walls())

    @staticmethod
    def _positions(sprites):
        return tuple(
            sprite.get_sprite_pos()
            if hasattr(sprite, "get_sprite_pos")
            else (sprite.x(), sprite.y())
            for sprite in sprites
        )

    def observation(self):
        """Return factual game state without choosing a reinforcement reward."""
        pacman = self._screen_field_mapper.get_pacman()
        ghosts = self._screen_field_mapper.get_ghosts()
        bonus_groups = (
            self._screen_field_mapper.get_straw(),
            self._screen_field_mapper.get_rasp(),
            self._screen_field_mapper.get_lemon(),
            self._screen_field_mapper.get_pear(),
        )
        return {
            "pacman": pacman.get_sprite_pos(),
            "ghosts": self._positions(ghosts),
            "walls": self._positions(self._screen_field_mapper.get_walls()),
            "coins": self._positions(self._screen_field_mapper.get_coins()),
            "points": self._positions(self._screen_field_mapper.get_points()),
            "bonuses": tuple(
                position
                for group in bonus_groups
                for position in self._positions(group)
            ),
            "coins_remaining": len(self._screen_field_mapper.get_coins()),
            "points_remaining": len(self._screen_field_mapper.get_points()),
            "score": self._controller.score(),
            "health": self._controller.health(),
            "screen_size": (
                self._screen_field_mapper.get_screen_width(),
                self._screen_field_mapper.get_screen_height(),
            ),
        }

    def advance(self, action=None, render=True, process_events=True):
        """Advance exactly one frame and expose terminal facts to an adapter."""
        active = self._process_events() if process_events else True

        if render:
            self._drawer.clear()

        if active:
            if self._mode == Mode.AUTO:
                self._controller.auto_game()
            else:
                self._player_game(action=action)
            self._check_statuses()

        victory = self._screen_field_mapper.is_victory()
        game_over = self._controller.is_gameover()

        if render:
            if victory:
                self._drawer.draw_victory()
            elif game_over:
                self._drawer.draw_gameover(self._controller.score())
            else:
                self._drawer.draw()
            self._drawer.update()

        info = {
            "victory": victory,
            "game_over": game_over,
            "window_closed": not active,
        }
        return self.observation(), victory or game_over, info

    def run(self, max_frames=None):
        active = True
        terminal = False
        frame_count = 0
        while active:
            if consts.LEARNING:
                self._clock.tick(100)
            else:
                self._clock.tick(100)

            if terminal:
                active = self._process_events()
            else:
                _, terminal, info = self.advance()
                active = not info["window_closed"]

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                active = False

        final_observation = self.observation()
        pygame.quit()
        return final_observation
