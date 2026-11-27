# Lancer PAC-AI ARENA sous Windows

## Démarrage rapide

Pour récupérer cette version depuis GitHub, clone la branche `ai/models-v1`. Les quatre checkpoints du lobby sont livrés avec le code.

```powershell
git clone --branch ai/models-v1 https://github.com/mkandr2004/pacman-rl.git
Set-Location pacman-rl
```

Ouvre PowerShell à la racine du dépôt local, dans le dossier qui contient `main/`, `maps/` et `models/`. Avec l’environnement local déjà installé, lance le lobby.

```powershell
.\.venv\Scripts\python.exe -m main.run
```

Pour créer un environnement neuf avec Python 3.14 et activer tous les choix du lobby, installe les dépendances du projet puis exécute la même commande. Le fichier `requirements-train.txt` contient aussi les bibliothèques requises pour charger les modèles en jeu.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -m main.run
```

Les quatre checkpoints du lobby sont des fichiers locaux dans `models/`. Si le lobby indique « Mode IA indisponible », le terminal précise l’erreur. Vérifie le chemin et les dépendances ou utilise l’archive autonome décrite plus bas.

## Choisir les deux contrôleurs

Le premier écran sélectionne Pac-Man. Le second sélectionne les fantômes, indépendamment du premier choix. La souris, les touches `1` à `3` et les flèches suivies d’Entrée fonctionnent dans les menus.

| Touche | Premier écran, Pac-Man | Second écran, fantômes |
|---|---|---|
| `1` | DAgger 42 | Sept fantômes classiques |
| `2` | Hybride 337 | Fantôme lent PPO 227 et six classiques |
| `3` | Joueur au clavier | Équipe PPO, deux fantômes appris et cinq classiques |

Les flèches déplacent Pac-Man quand tu joues au clavier. Dans une partie lancée depuis le lobby, Échap revient au lobby. Dans les menus, cette touche revient à l’écran précédent ou ferme le jeu depuis le premier écran. Sur l’écran de résultat d’une partie avec IA, Entrée relance la même configuration et Échap revient au lobby. Pour ouvrir directement le jeu clavier avec les fantômes classiques, ajoute `--keyboard` à la commande de lancement. Dans ce mode direct, ferme la fenêtre pour quitter.

## Préparer une démonstration

Le paramètre `--seed` fixe la graine des parties avec IA. Cette commande prépare le duel contre l’équipe PPO. Pour une autre démonstration, remplace `95817` par la graine correspondante dans le tableau. Dans le premier écran, choisis `1` ou `2`. Les issues ci-dessous ont été vérifiées avec les deux modèles Pac-Man.

```powershell
.\.venv\Scripts\python.exe -m main.run --seed 95817
```

| Graine | Touche au second écran | Issue attendue |
|---:|---|---|
| 92000 | `1` · Classiques | Victoire de Pac-Man, 136 points |
| 95207 | `2` · Fantôme lent PPO 227 | Victoire des fantômes |
| 95817 | `3` · Équipe PPO | Victoire des fantômes avec une capture du rapide appris |

Les nombres de victoires affichés au premier écran du lobby correspondent au benchmark scientifique contre les fantômes classiques. Le jeu vérifie l’empreinte SHA-256 des quatre checkpoints au chargement.

## Utiliser les archives autonomes

Dans la copie de travail d’origine, les trois archives locales sont des dossiers voisins du dépôt. Elles contiennent chacune leur interpréteur Python Windows et leurs dépendances. Pour retrouver le lobby et ses quatre modèles depuis le secours local, ouvre `pacman-rl-duel-deux-2026-11-27` dans PowerShell.

```powershell
.\Voir-Duel.ps1
```

L’archive du duel démarre sur la graine 95817. Pour faire la démonstration des classiques ou du lent PPO 227, appelle le lanceur avec `-Seed 92000` ou `-Seed 95207`, puis sélectionne le mode correspondant dans le lobby.

| Archive | Documentation | Contenu |
|---|---|---|
| `pacman-rl-duel-deux-2026-11-27` | `README.md` | Lobby complet et quatre modèles fonctionnels |
| `pacman-rl-legacy-2026-11-24` | `README_COMPATIBILITE.md` | DAgger 42 de secours, checkpoints et commandes historiques |
| `pacman-rl-hybride337-2026-11-26` | `README.md` | Hybride 337, configuration et démonstrations archivées |

## Retrouver les résultats scientifiques

La copie de travail d’origine conserve dans `runs/` les résultats détaillés, configurations, courbes et checkpoints expérimentaux, y compris les essais négatifs. Les synthèses locales `docs/SYNTHESE_SCIENTIFIQUE_PACMAN.md` et `docs/SYNTHESE_MARL_2026-11-27.md` décrivent les protocoles, les résultats et leurs limites.

Les neuf combinaisons du lobby ont été vérifiées dans le projet et dans l’archive du duel avec le pilote de test SDL. Les deux archives Pac-Man ont aussi reproduit une victoire à 136 points avec leurs propres interpréteurs. Les preuves sont dans `runs/verification_lobby_20261127/` sur la copie de travail d’origine.
