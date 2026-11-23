# PAC-AI ARENA

PAC-AI ARENA est un projet academique visant a construire un jeu de grille inspire de Pac-Man et a comparer des agents aleatoires, heuristiques et d'apprentissage par renforcement.

## Etat du projet

Le projet integre maintenant un moteur de jeu Pygame visuel comme base de travail. L'interface pour les agents, les observations, les recompenses et l'entrainement RL seront ajoutes progressivement et testes separement.

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

## Moteur graphique tiers

Le moteur graphique initial est adapte de [`StepanTita/pacman`](https://github.com/StepanTita/pacman), importe depuis le commit `bbab75ed72dcbd6ce224aa3e718a2875afd57656` sous licence MIT.

La licence et le detail des composants importes sont conserves dans [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) et [`LICENSES/StepanTita-pacman-MIT.md`](LICENSES/StepanTita-pacman-MIT.md). Les futurs commits identifieront separement les adaptations realisees pour PAC-AI ARENA.

## Lancer le jeu visuel

Sous Windows avec Python 3.14 :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m main.run
```

Le jeu s'ouvre dans une fenetre Pygame. Pac-Man se controle avec les touches directionnelles.

Le fichier `requirements-upstream.txt` conserve les versions historiques du depot tiers. PAC-AI ARENA utilise `pygame-ce` localement car il fournit un wheel compatible avec Python 3.14 tout en conservant l'API `import pygame` utilisee par le moteur.

## Controle externe local

La demonstration suivante remplace temporairement le clavier par une politique scriptée :

```powershell
.\.venv\Scripts\python.exe -m scripts.scripted_visual_demo
```

Cette politique repete des directions pour verifier que le moteur visuel accepte une action externe. **Ce n'est pas un agent RL entraine.** Le futur environnement Gymnasium reutilisera la meme interface apres definition et validation des observations, recompenses et terminaisons.

Les tests locaux verifient que le mode clavier demarre toujours et qu'une politique externe peut deplacer Pac-Man :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Premier environnement RL local

`rl.pacman_env.PacmanEnv` expose le moteur avec l'API Gymnasium. Cette premiere version entraine uniquement Pac-Man ; les ghosts conservent leurs comportements programmes.

Les quatre actions discretes sont :

| Action | Direction |
|---:|---|
| 0 | gauche |
| 1 | droite |
| 2 | haut |
| 3 | bas |

L'observation contient six canaux de grille (murs, pieces, points, bonus, Pac-Man et ghosts), les positions exactes des personnages et la vie restante. La grille explique la structure du niveau ; les positions exactes evitent de perdre les mouvements effectues a l'interieur d'une case.

La recompense initiale est volontairement decomposable :

- `-0.01` par decision ;
- variation positive du score pour les objets collectes ;
- `-25` par vie perdue ;
- `+100` pour une victoire ;
- `-100` pour un game over.

Ces valeurs sont des hypotheses de depart, pas des resultats scientifiques. Chaque composante est retournee dans `info["reward_terms"]` afin de pouvoir diagnostiquer et comparer les variantes sans modifier les regles du jeu.

Une politique aleatoire permet de valider un episode complet avant d'ajouter un algorithme d'apprentissage :

```powershell
.\.venv\Scripts\python.exe -m scripts.random_baseline --episodes 3 --max-steps 300
```

Cette baseline n'est pas un agent entraine. Elle verifie les observations, les actions, les recompenses, les terminaisons et la reproductibilite par seed.
