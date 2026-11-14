# PAC-AI ARENA

PAC-AI ARENA est un projet academique visant a construire un jeu de grille inspire de Pac-Man et a comparer des agents aleatoires, heuristiques et d'apprentissage par renforcement.

## Etat du projet

Le projet est en **phase 0 : audit et choix d'architecture**. Aucun moteur de jeu ni agent RL n'est encore implemente. Les choix techniques restent soumis a validation avant chaque phase.

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
