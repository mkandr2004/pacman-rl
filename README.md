# PAC-AI ARENA

PAC-AI ARENA est un projet academique visant a construire un jeu de grille inspire de Pac-Man et a comparer des agents aleatoires, heuristiques et d'apprentissage par renforcement.

## Etat du projet

La **phase 1 : noyau deterministe** est implementee sur la branche `phase-01/core`. Le moteur applique les actions de Pac-Man et du fantome sans affichage, et les regles sont couvertes par des tests. Aucun rendu Pygame, agent heuristique ou agent RL n'est encore implemente.

## Objectif scientifique

Construire une chaine reproductible : environnement deterministe, baselines, entrainement, sauvegarde, evaluation et demonstration de deux politiques distinctes dans une arene IA contre IA.

## Principes

- separer les regles du jeu du rendu ;
- tester les regles sans interface graphique ;
- distinguer score du jeu, recompense RL et resultat d'un match ;
- comparer les agents aux memes baselines et seeds d'evaluation ;
- documenter honnetement les limites et les resultats.

## Roadmap

Audit -> noyau deterministe -> rendu -> baselines -> agents RL -> arene -> experiences -> soutenance.

Chaque etape est developpee, testee et validee avant de passer a la suivante.

## Installation de developpement

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Verifications de la phase 1

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\demo_core.py
```

Les regles exactes du tick sont decrites dans [`docs/game-rules.md`](docs/game-rules.md).
