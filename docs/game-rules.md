# Regles du noyau deterministe

Ce document decrit les regles implementees en phase 1. Il ne decrit ni une politique heuristique, ni une recompense d'apprentissage par renforcement.

## Carte

Une carte ASCII utilise cinq symboles :

| Symbole | Signification |
|---|---|
| `#` | Mur infranchissable |
| `P` | Position initiale de Pac-Man |
| `G` | Position initiale du fantome |
| `.` | Pastille |
| espace | Case vide |

La carte doit etre rectangulaire et contenir exactement un Pac-Man, un fantome et au moins une pastille.

## Actions

Les deux acteurs choisissent une direction parmi `UP`, `DOWN`, `LEFT` et `RIGHT`. Il n'existe pas de `no-op` dans cette version : cette action augmenterait l'espace de recherche et pourrait encourager l'inaction sans etre necessaire aux regles actuelles.

Une action dirigee vers un mur ou hors de la grille laisse l'acteur sur sa case et produit un evenement `*_blocked`.

## Resolution d'un tick

Un tick suit toujours cet ordre :

1. calculer les deux destinations depuis le meme etat initial ;
2. bloquer independamment les mouvements invalides ;
3. deplacer simultanement Pac-Man et le fantome ;
4. collecter une eventuelle pastille sur la destination de Pac-Man ;
5. detecter une capture ;
6. choisir l'unique resultat de l'episode selon les priorites ci-dessous.

Les actions ne sont pas appliquees l'une apres l'autre. Une resolution sequentielle avantagerait artificiellement l'acteur joue en premier.

## Capture

Le fantome capture Pac-Man dans deux cas :

- les deux acteurs terminent le tick sur la meme case ;
- ils echangent leurs positions pendant le meme tick.

Le second cas evite que les acteurs se traversent sans collision.

## Resultats et priorites

Les resultats sont mutuellement exclusifs :

1. `ghost_win` si une capture a lieu ;
2. `pacman_win` si toutes les pastilles ont ete mangees sans capture ;
3. `timeout` si le plafond de pas est atteint sans victoire ni capture ;
4. `in_progress` sinon.

La capture est prioritaire lorsque Pac-Man mange la derniere pastille sur le meme tick. La pastille est tout de meme comptee dans le score, car Pac-Man a atteint sa case, mais le resultat du match est `ghost_win`.

Un timeout est un match nul distinct : il n'est transforme ni en victoire de Pac-Man ni en victoire du fantome.

## Etat, score et evenements

`GameState` contient uniquement ce qui est necessaire pour continuer la partie : positions, pastilles restantes, nombre de pas, score et resultat.

Le score augmente d'un point par pastille. Le noyau ne calcule aucune recompense RL. Il produit des evenements factuels comme `pellet_eaten`, `capture` ou `timeout` ; un futur adaptateur RL pourra convertir ces evenements en recompense sans modifier les regles du jeu.

## Determinisme et seed

La carte de phase 1 est fixe : meme etat et memes actions donnent toujours la meme transition. `reset(seed=...)` accepte deja une seed pour stabiliser l'interface, mais celle-ci ne change pas encore la carte. Toute future randomisation devra utiliser cette seed et conserver les tests de reproductibilite.
