# Diagnostic des fantômes et comparaison des deux Pac-Man

Les deux checkpoints du lobby jouent sur la carte originale. Pour chaque configuration, le script rejoue les mêmes graines sans affichage et reprend les règles de décision du lobby. Les deux Pac-Man choisissent leurs actions de façon déterministe et les fantômes PPO échantillonnent leurs actions avec une graine fixée.

Depuis la racine du dépôt, avec les dépendances de `requirements-train.txt` installées dans l’environnement Python du jeu, exécuter cette commande.

```powershell
python -m scripts.diagnose_ghosts --seeds 92000 92001 92002 92003 92004 92005 92006 95207 95817 --output runs/ghost-diagnostics.json
```

Le rapport JSON contient les résultats par partie, les 30 premières positions de Pac-Man, sa plus longue séquence sans collecte et les mesures par fantôme. Une décision couvre jusqu’à quatre frames de simulation. Le rapport relève les décisions où le fantôme ne bouge à aucune frame, les retours à sa position d’il y a deux décisions et, pour les fantômes contrôlés, les frames sans déplacement après une commande directionnelle. Dans les modes PPO, le masque permet `WAIT` uniquement lorsqu’aucune direction n’est disponible.

Choisir un nouveau chemin de sortie pour chaque exécution afin de conserver les rapports précédents.

## Résultats exploratoires

Chaque cellule donne le nombre de victoires de Pac-Man sur neuf parties. Les trois graines de démonstration sont incluses dans l’échantillon.

| Pac-Man | Classiques | Fantôme lent PPO | Équipe PPO | Poursuite et interception |
|---|---:|---:|---:|---:|
| DAgger 42 | 9 | 3 | 4 | 1 |
| Hybride 337 | 9 | 4 | 3 | 1 |

Les quatre configurations se terminent toutes par une victoire de Pac-Man ou des fantômes. L’équipe de poursuite et interception garde les cinq fantômes classiques et utilise, pour les deux autres, une recherche en largeur sur la grille et le même masque physique que l’équipe PPO. Le lent vise Pac-Man. Le rapide vise une intersection située jusqu’à quatre cases devant Pac-Man, ou la dernière case libre de cette projection. Les deux fantômes programmés retirent **2 vies par partie en moyenne** contre **1** pour les deux fantômes de l’équipe PPO, agrégé sur les deux Pac-Man.

La seule immobilisation sur une décision entière concerne le mutant classique nº 5 à la dernière décision de la partie (527), en `(360, 600)`, face à DAgger 42 et avec l’équipe PPO, graine `92006`. Sa dernière direction est le bas et les directions physiques disponibles à la fin de la décision sont la droite et le haut. Le blocage prolongé signalé reste à reproduire.

Les fantômes contrôlés de l’équipe PPO reviennent à leur position d’il y a deux décisions dans 36,5 % des décisions admissibles pour cette mesure, contre 15,8 % pour leurs homologues programmés. Ces allers-retours peuvent donner l’impression d’un fantôme bloqué dans un couloir.

Face aux classiques, les 18 parties partagent exactement les mêmes 30 premières positions de Pac-Man. Face à l’équipe PPO, DAgger 42 montre trois ouvertures distinctes sur neuf parties et Hybride 337 une seule. La plus longue séquence sans collecte atteint en moyenne 62 décisions pour DAgger 42 et 84 pour Hybride 337 face à cette équipe.

Cette stratégie sert de référence dans le script. Elle a été évaluée sur une carte et un petit ensemble de graines. Pour la soutenance, DAgger 42 illustre l’imitation de l’expert sur les états visités par l’agent. Hybride 337 combine cette imitation, un critique préentraîné et un affinage PPO. Les résultats sur les deux adversaires PPO s’inversent d’un modèle à l’autre sur cet échantillon.
