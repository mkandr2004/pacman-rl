"""Pygame entry point for selecting a local Pac-Man controller."""

from pathlib import Path
import secrets
import sys

import pygame

from main.arena_models import ARENA_MODELS, GHOST_MODEL, TEAM_MODEL, load_model


SIZE = (960, 720)
BACKGROUND = "#0b1120"
PANEL = "#141e31"
LINE = "#30405b"
TEXT = "#f5f8ff"
SECONDARY = "#b6c3dd"
GOLD = "#ffdc63"
CYAN = "#73e5de"
CARDS = (pygame.Rect(54, 249, 416, 222), pygame.Rect(490, 249, 416, 222),
         pygame.Rect(54, 493, 852, 78))
CHOICES = (*ARENA_MODELS, None)
KEY_CHOICES = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2,
               pygame.K_KP1: 0, pygame.K_KP2: 1, pygame.K_KP3: 2}


class Screen:
    def __init__(self, surface):
        self.surface = surface
        self.fonts = {}

    def text(self, text, x, y, size, color=TEXT, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont("segoeui", size, bold=bold)
        self.surface.blit(self.fonts[key].render(text, True, color), (x, y))

    def panel(self, rect, outline=LINE):
        pygame.draw.rect(self.surface, PANEL, rect, border_radius=18)
        pygame.draw.rect(self.surface, outline, rect, width=2, border_radius=18)


def draw_lobby(screen, selected):
    screen.surface.fill(BACKGROUND)
    screen.text("PAC-AI ARENA", 54, 39, 26, GOLD, bold=True)
    pygame.draw.line(screen.surface, LINE, (54, 87), (906, 87), 2)
    screen.text("Choisir Pac-Man", 54, 111, 58, bold=True)
    screen.text("Labyrinthe original · Sept fantômes", 56, 197, 23, SECONDARY)

    for i, (rect, model, accent) in enumerate(zip(CARDS[:2], ARENA_MODELS, (GOLD, CYAN))):
        screen.panel(rect, accent if i == selected else LINE)
        x = rect.x + 24
        screen.text(str(i + 1), x, 271, 18, accent, bold=True)
        screen.text(model.name, x, 302, 34, bold=True)
        screen.text(model.description, x, 352, 20, SECONDARY)
        screen.text(f"{model.wins} / 100", x, 399, 32, accent, bold=True)
        screen.text("contre les classiques", x + 158, 411, 18, SECONDARY)

    screen.panel(CARDS[2], GOLD if selected == 2 else LINE)
    screen.text("3", 78, 518, 20, GOLD if selected == 2 else SECONDARY, bold=True)
    screen.text("Jouer au clavier", 116, 513, 26, bold=True)
    screen.text("Touches directionnelles", 633, 520, 19, SECONDARY)
    pygame.draw.line(screen.surface, LINE, (54, 618), (906, 618), 2)
    screen.text("1–3  Sélection directe     Flèches + Entrée  Choisir     Échap  Quitter", 55, 644, 21, SECONDARY)
    pygame.display.flip()


def choose_cards(surface, cards, draw, back=None):
    screen = Screen(surface)
    selected = 0
    clock = pygame.time.Clock()
    while True:
        draw(screen, selected)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return back
                if event.key in KEY_CHOICES and KEY_CHOICES[event.key] < len(cards):
                    return KEY_CHOICES[event.key]
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return selected
                if event.key in (pygame.K_LEFT, pygame.K_UP):
                    selected = (selected - 1) % len(cards)
                elif event.key in (pygame.K_RIGHT, pygame.K_DOWN):
                    selected = (selected + 1) % len(cards)
            if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
                for i, rect in enumerate(cards):
                    if rect.collidepoint(event.pos):
                        selected = i
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                            return selected
        clock.tick(30)


def choose(surface):
    return choose_cards(surface, CARDS, draw_lobby)


def choose_ghosts(surface, pacman_name):
    cards = CARDS if TEAM_MODEL else CARDS[:2]

    def draw(screen, selected):
        screen.surface.fill(BACKGROUND)
        screen.text("PAC-AI ARENA", 54, 39, 26, GOLD, bold=True)
        pygame.draw.line(screen.surface, LINE, (54, 87), (906, 87), 2)
        screen.text("Choisir les fantômes", 54, 111, 58, bold=True)
        screen.text(f"Pac-Man · {pacman_name}", 56, 197, 23, SECONDARY)
        options = (("Classiques", "Sept contrôleurs du moteur", GOLD),
                   (GHOST_MODEL.name, "Un fantôme appris · six classiques", CYAN))
        for i, (rect, (name, description, accent)) in enumerate(zip(cards[:2], options, strict=True)):
            screen.panel(rect, accent if i == selected else LINE)
            screen.text(str(i + 1), rect.x + 24, 271, 18, accent, bold=True)
            screen.text(name, rect.x + 24, 313, 34, bold=True)
            screen.text(description, rect.x + 24, 374, 21, SECONDARY)
            if i == 1:
                screen.text("Prototype expérimental", rect.x + 24, 423, 17, accent)
        if TEAM_MODEL:
            screen.panel(CARDS[2], CYAN if selected == 2 else LINE)
            screen.text("3", 78, 516, 20, CYAN, bold=True)
            screen.text(TEAM_MODEL.name, 116, 512, 26, bold=True)
            screen.text(f"Prototype · {TEAM_MODEL.count} appris · {7-TEAM_MODEL.count} classiques",
                        525, 520, 19, SECONDARY)
        pygame.draw.line(screen.surface, LINE, (54, 618), (906, 618), 2)
        screen.text(f"1–{len(cards)}  Sélection directe     Flèches + Entrée  Choisir     Échap  Retour",
                    55, 644, 21, SECONDARY)
        pygame.display.flip()

    return choose_cards(surface, cards, draw, back="back")


def message(title, detail, hint=""):
    surface = pygame.display.get_surface()
    screen = Screen(surface)
    surface.fill(BACKGROUND)
    x, y = (surface.get_width() - 788) // 2, (surface.get_height() - 234) // 2
    screen.panel(pygame.Rect(x, y, 788, 234), GOLD)
    screen.text(title, x + 32, y + 34, 48, bold=True)
    screen.text(detail, x + 34, y + 113, 26, SECONDARY)
    if hint:
        screen.text(hint, x + 34, y + 177, 20, GOLD)
    pygame.display.flip()


def wait_after_message(replay=False):
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "lobby"
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return "replay" if replay else "lobby"


def play_model(entry, model, seed=None, *, ghost_entry=None, ghost_model=None):
    duel = ghost_model is not None
    deterministic = not duel or not ghost_entry.stochastic
    pacman_name = entry.name if entry is not None else "Clavier"
    name = pacman_name
    if duel:
        if ghost_entry.count > 1:
            from rl.ghost_team_env import GhostTeamEnv

            env = GhostTeamEnv(model, ghost_entry.count, render_mode="human", keyboard=model is None)
        else:
            from rl.ghost_env import SlowGhostEnv

            env = SlowGhostEnv(model, render_mode="human", keyboard=model is None)
        base = env.base
        controller = ghost_model
        masks = env.action_masks
        name = f"{pacman_name} · {ghost_entry.name}"
    else:
        from rl.imitation_env import make_imitation_env

        env = make_imitation_env(danger=False, render_mode="human")
        base = env.unwrapped
        controller = model
        masks = base.action_masks
    try:
        while True:
            match_seed = secrets.randbits(32) if seed is None else seed
            if not deterministic:
                from rl.ghost_evaluation import seed_sampling

                seed_sampling(match_seed)
            observation, _ = env.reset(seed=match_seed)
            base._game.enable_lobby_return()
            pygame.display.set_caption(f"PAC-AI ARENA | {name} | Échap : lobby")
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = controller.predict(observation, deterministic=deterministic, action_masks=masks())
                observation, _, terminated, truncated, info = env.step(action)
            if info["window_closed"]:
                return "lobby" if base._game.lobby_requested else "quit"
            title = "Victoire" if info["victory"] else "Défaite" if info["game_over"] else "Temps écoulé"
            if duel and info["victory"]:
                title = "Victoire de Pac-Man"
            elif duel and info["ghost_win"]:
                title = "Victoire des fantômes"
            lives = "vie" if info["health"] <= 1 else "vies"
            detail = f"{name}   ·   {info['score']} points   ·   {info['health']} {lives}"
            message(title, detail, "Entrée  Rejouer     Échap  Revenir au lobby")
            route = wait_after_message(replay=True)
            if route != "replay":
                return route
    finally:
        env.close()


def run_lobby(create_game, seed=None):
    # The imported engine resolves its assets relative to the checkout root.
    if not Path("maps/map_2.txt").is_file():
        raise FileNotFoundError("Lancez PAC-AI ARENA depuis la racine du dépôt.")
    try:
        while True:
            pygame.init()
            surface = pygame.display.set_mode(SIZE)
            pygame.display.set_caption("PAC-AI ARENA | Lobby")
            selected = choose(surface)
            if selected is None:
                return
            entry = CHOICES[selected]
            ghost_choice = choose_ghosts(surface, entry.name if entry is not None else "Clavier")
            if ghost_choice is None:
                return
            if ghost_choice == "back":
                continue
            ghost_entry = (None, GHOST_MODEL, TEAM_MODEL)[ghost_choice]
            if entry is None and ghost_entry is None:
                game = create_game()
                game.enable_lobby_return()
                pygame.display.set_caption("PAC-AI ARENA | Clavier | Échap : lobby")
                game.run()
                if not game.lobby_requested:
                    return
                continue
            message("Chargement de la partie", "Ouverture des modèles locaux")
            try:
                model = load_model(entry) if entry is not None else None
                ghost_model = load_model(ghost_entry) if ghost_entry is not None else None
                route = play_model(entry, model, seed, ghost_entry=ghost_entry, ghost_model=ghost_model)
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                print(f"Mode IA: {error}", file=sys.stderr)
                pygame.init()
                pygame.display.set_mode(SIZE)
                message("Mode IA indisponible", "Vérifiez les poids locaux et les dépendances IA.",
                        "Entrée ou Échap  Revenir au lobby")
                route = wait_after_message()
            if route == "quit":
                return
    finally:
        pygame.quit()
