# PAC-AI ARENA

PAC-AI ARENA est un projet academique visant a construire un jeu de grille inspire de Pac-Man et a comparer des agents aleatoires, heuristiques et d'apprentissage par renforcement.

## Etat du projet

Le moteur Pygame permet de jouer au clavier et de regarder DAgger 42 ou Hybride 337 jouer sur la carte originale. Le lobby propose les fantômes classiques, le lent PPO 227 et l’équipe de deux fantômes à politique partagée. Les entraînements de Pac-Man et des fantômes sont clôturés. La suite du projet porte sur le jeu, l’interface et la présentation. [Guide de lancement](docs/LANCER_LE_JEU.md)

## Modèles v1

Les quatre checkpoints du lobby sont inclus dans `models/`. Le lobby vérifie leur empreinte SHA-256 au chargement.

| Contrôleur | Modèle | Validation |
|---|---|---|
| Pac-Man | DAgger 42 | 98 victoires sur 100 contre les fantômes classiques |
| Pac-Man | Hybride 337 | 97 victoires sur 100 sur le même benchmark |
| Fantômes | PPO 227 | Un fantôme lent appris et six classiques |
| Fantômes | Équipe PPO | Deux fantômes à politique partagée et cinq classiques |

Les checkpoints expérimentaux et les résultats détaillés des campagnes sont conservés localement.

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

Depuis la racine du dépôt, avec l’environnement local déjà installé, lance le lobby.

```powershell
.\.venv\Scripts\python.exe -m main.run
```

Choisis d’abord Pac-Man, puis les fantômes, avec la souris ou les touches `1` à `3`. Le guide détaille l’installation, les commandes de démonstration et les trois archives autonomes. [Guide de lancement sous Windows](docs/LANCER_LE_JEU.md)

Le fichier `requirements-upstream.txt` conserve les versions historiques du depot tiers. PAC-AI ARENA utilise `pygame-ce` localement car il fournit un wheel compatible avec Python 3.14 tout en conservant l'API `import pygame` utilisee par le moteur.

## Historique expérimental

Les sections suivantes décrivent les étapes d’apprentissage et les commandes conservées pour la reproduction scientifique. Le jeu v1 utilise les modèles fournis.

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

## Baseline heuristique locale

La politique heuristique utilise une recherche en largeur (BFS) : elle cherche le plus court chemin vers un objet, evite les murs et considere les cases proches des ghosts comme dangereuses. Elle ne contient aucun apprentissage.

```powershell
.\.venv\Scripts\python.exe -m scripts.heuristic_baseline --episodes 1 --max-steps 500 --render
.\.venv\Scripts\python.exe -m scripts.compare_baselines --episodes 5 --max-steps 500
```

Cette baseline sert de seuil plus exigeant que l'agent aleatoire : un modele entraine doit idealement finir par la depasser.

## Premier entrainement DQN local

Les dependances lourdes d'entrainement restent separees de celles du jeu :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

Entrainer puis evaluer un modele :

```powershell
.\.venv\Scripts\python.exe -m scripts.train_dqn --timesteps 10000 --max-steps 500 --eval-episodes 5
.\.venv\Scripts\python.exe -m scripts.evaluate_dqn --model CHEMIN_DU_MODELE_AFFICHE.zip --episodes 1 --max-steps 500 --render
```

Les positions en pixels sont normalisees entre `0` et `1` uniquement pour le reseau neuronal. La grille symbolique et les recompenses ne changent pas. Les checkpoints expérimentaux et les rapports dans `runs/` sont ignorés par Git.

Premier resultat local, seed d'entrainement `42`, cinq seeds d'evaluation et 500 decisions maximum par episode :

| Politique | Entrainement | Score moyen | Recompense moyenne | Victoires |
|---|---:|---:|---:|---:|
| Aleatoire | aucun | 12,8 | -17,2 | 0/5 |
| Heuristique BFS | aucun | 79,0 | 49,0 | 0/5 |
| DQN | 10 000 pas | 24,2 | -5,8 | 0/5 |

Ces premiers chiffres correspondent au prototype avec des bonus mesures en temps reel. Ils sont conserves comme historique technique et ne doivent pas etre compares aux nouvelles experiences utilisant l'horloge de simulation.

## Experiences reproductibles avec horloge de simulation

Dans l'environnement RL, une frame represente maintenant exactement 10 ms de simulation, avec ou sans affichage. Une decision repete une direction pendant quatre frames, soit 40 ms. La vitesse de calcul ne modifie plus la duree des bonus ou de l'invincibilite. Le mode clavier conserve son horloge temps reel.

Le hasard des ghosts est propre a chaque environnement et ne consomme plus celui du programme d'apprentissage. Les seeds d'entrainement sont distinctes des seeds d'evaluation.

Lancer trois entrainements de 30 000 decisions et recalculer les baselines sur les memes seeds d'evaluation :

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiments --timesteps 30000 --seeds 42 43 44 --max-steps 500
```

Un nouveau dossier date est cree sous `runs/`. Il contient tous les modeles (un par seed), les fichiers `training.monitor.csv` avec les recompenses et scores de chaque episode, les rapports individuels, `summary.json` et `learning_curves.png`. Le rapport conserve les hyperparametres, les versions des bibliotheques et les seeds.

Les courbes montrent une moyenne glissante sur dix episodes d'entrainement. Les scores finaux proviennent de cinq episodes d'evaluation distincts (seeds 1000 a 1004), avec exploration desactivee. Une augmentation de recompense a l'entrainement ne suffit donc pas a annoncer une amelioration sur l'evaluation.

Pour rejouer un modele, remplacer le chemin par celui affiche a la fin de l'experience :

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_dqn --model runs/experiment_DATE_HEURE/dqn_seed_42.zip --episodes 1 --max-steps 500 --render
```

Une execution individuelle de `train_dqn` cree par defaut un dossier date sous `runs/` et y sauvegarde le modele. Un `--run-dir` explicite doit etre inexistant. Le script refuse d'ecraser un modele deja present. L'evaluation demande un chemin `--model` explicite pour choisir le checkpoint a rejouer.

Campagne locale terminee le 23 novembre 2026 : trois seeds d'entrainement, 30 000 decisions chacune, episodes limites a 500 decisions, cinq seeds d'evaluation identiques par modele. Ces chiffres utilisent tous l'horloge de simulation corrigee :

| Politique | Seed d'entrainement | Score moyen | Recompense moyenne | Victoires |
|---|---:|---:|---:|---:|
| Aleatoire | aucune | 4,2 | -172,76 | 0/5 |
| BFS | aucune | 79,0 | -99,17 | 0/5 |
| DQN | 42 | 5,6 | -171,46 | 0/5 |
| DQN | 43 | 22,6 | -154,52 | 0/5 |
| DQN | 44 | 30,8 | -116,93 | 0/5 |

Le DQN obtient un score moyen de 19,67 sur les trois seeds, avec une forte variabilite. Aucun des trois modeles ne depasse BFS. Il s'agit d'un bilan exploratoire (cinq episodes par modele), pas d'une validation finale ou d'une preuve de generalisation a d'autres cartes. Les courbes et rapports complets sont conserves localement dans `runs/experiment_20261123_155548/`.

## PPO sur le niveau original (observations v3)

L'experience PPO conserve `maps/map_2.txt`, les sept ghosts et tous les bonus. Trente pastilles se trouvent dans une zone separee par des murs : l'agent doit aussi apprendre a utiliser le casse-mur. Aucune victoire sur une carte simplifiee n'est comptee comme une victoire sur ce niveau.

L'observation v3 ajoute les quatre types de bonus et la duree restante des effets (invincibilite, vitesse, casse-mur). La vitesse est recalculee depuis la vitesse de base quand un bonus commence ou expire ; des bonus repetes ne peuvent plus accumuler des divisions entieres jusqu'a immobiliser Pac-Man.

MaskablePPO choisit parmi les quatre directions. Le masque retire seulement les mouvements physiquement bloques au prochain pas ; le casse-mur autorise la traversee des murs. Il ne choisit aucun chemin et ne cache pas les ghosts. En cas d'impasse totale, les quatre actions restent definies et les collisions normales s'appliquent.

Le reseau comprend deux couches de 256 neurones pour la politique et deux pour la valeur, soit 1 129 221 parametres. Deux simulations tournent dans des processus separes. La configuration initiale utilise gamma 0,995, learning rate 0,0003, batches de 256 et dix epochs par collecte. La recompense n'est pas modifiee. La limite d'episode passe de 500 a 3000 decisions pour permettre de parcourir toute la carte ; les anciens scores a 500 decisions ne sont donc pas directement comparables.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -m scripts.train_ppo --timesteps 262144 --workers 2 --episodes 20
```

Les fichiers sont sauvegardes dans un nouveau dossier `runs/ppo_DATE_HEURE/` : configuration, logs Monitor, checkpoints intermediaires, dernier modele et modele selectionne par la recompense de validation. L'evaluation finale utilise les seeds 1000 a 1019, distinctes de la validation initialisee avec 2000.

Pour visualiser un checkpoint PPO, remplacer le chemin par celui de son dossier :

```powershell
.\.venv\Scripts\python.exe -m scripts.train_ppo --model runs/ppo_DATE_HEURE/best_model.zip --episodes 1 --render
```

Les modeles DQN des observations v2 et les modeles PPO des observations v3 correspondent a des contrats differents. Un nombre de parametres plus eleve ne garantit pas un taux de victoire eleve : le rapport final reste la mesure de reference.

Resultat local du 24 novembre 2026, dossier `runs/ppo_20261124_020053/` : campagne de 262 144 decisions ; checkpoint selectionne sur validation a 196 608 decisions. Evaluation finale sur 20 seeds, avec un plafond de 3000 decisions :

| Politique | Score moyen | Victoires | Defaites | Timeouts |
|---|---:|---:|---:|---:|
| Aleatoire | 4,4 | 0/20 | 20/20 | 0/20 |
| BFS | 79,0 | 0/20 | 20/20 | 0/20 |
| PPO selectionne | 32,0 | 0/20 | 0/20 | 20/20 |

**Evaluation historique invalidee par l'audit des bordures.** Ces timeouts et la recompense de 2,0 provenaient d'une sortie de carte apres destruction d'un mur exterieur, et ne prouvaient pas une capacite de survie. Sur la seed 1000, la bordure en (540, 660) etait detruite a la frame 573 ; le rectangle de Pac-Man sortait de l'ecran a la frame 618, sans terminaison d'episode. **L'objectif d'un agent gagnant n'est pas atteint.**

Le dernier modele (`final_model.zip`), le meilleur de validation (`best_model.zip`), les checkpoints reguliers et la courbe (`validation_curve.png`) restent disponibles localement. Commande de demonstration :

```powershell
.\.venv\Scripts\python.exe -m scripts.train_ppo --model runs/ppo_20261124_020053/best_model.zip --episodes 1 --render
```

## Correction locale des bordures et reprise du PPO (v4)

Le moteur protege desormais les quatre bordures, avec ou sans citron, et refuse chaque deplacement hors ecran, y compris lors des repetitions d'action. Le mode clavier utilise le meme controle. Les murs interieurs restent cassables uniquement pendant l'effet du citron. La carte et les ghosts sont conserves.

Sur le moteur corrige, le checkpoint d'origine a ete rejoue sans apprentissage sur les seeds 1000, 7000, 7001, 7002 et 7003 : cinq defaites, 32 points de score par partie (12 petites pastilles et 4 pieces), aucune sortie et aucune bordure detruite.

| Terme de recompense RL | Avant | Reprise v4 |
|---|---:|---:|
| Petite pastille, une seule fois | +1 | +3 |
| Piece, une seule fois | +5 | +5 |
| Victoire (toutes les pastilles et pieces) | +100 | +500 |
| Vie perdue | -25 | -25 |
| Game over | -100 | -100 |
| Decision | -0,01 | -0,01 |
| Decision sans deplacement | 0 | -0,05 |
| Fruit ou mur detruit, directement | 0 | 0 |

Le score affiche reste Point=1, Coin=5. Aucune direction, case a casser ou trajectoire n'est imposee. Les dimensions d'observation et l'architecture du PPO sont inchangees : la reprise charge les poids existants et leur optimiseur. Le taux d'apprentissage passe a 0,0001 et l'entropie a 0,03 pour cette reprise.

La commande suivante sauvegarde d'abord le modele d'origine dans un nouveau dossier date, verifie son hash, entraine, selectionne sur validation puis audite les limites avant de mettre a jour **le meme fichier de modele**. La sauvegarde, les poids intermediaires et les resultats restent conserves.

```powershell
.\.venv\Scripts\python.exe -m scripts.resume_ppo --model runs/ppo_20261124_020053/best_model.zip --timesteps 65536 --workers 2
```

Pour executer seulement un audit sans apprentissage, utiliser un dossier de sortie inexistant :

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_policy --model runs/ppo_20261124_020053/best_model.zip --output runs/nouvel_audit --seeds 1000 7000 7001 7002 7003
```

Reprise locale executee le 24 novembre 2026 : `runs/resume_20261124_040031/`, 65 536 decisions supplementaires depuis les poids a 196 608 decisions. Aucun checkpoint n'a depasse la reference sur validation (-122,052 avec les recompenses v4 ; dernier checkpoint environ -136,42). Le fichier principal reste donc identique en poids a son original, plutot que de le remplacer par une regression.

La sauvegarde initiale verifiee se trouve dans `runs/resume_20261124_040031/original_model.zip` ; les nouveaux poids effectivement entraines restent dans `final_model.zip` du meme dossier. Le rapport de selection est dans `report.json`.

L'audit du checkpoint retenu sur cinq parties donne toujours 32 points (12 Points et 4 Coins), cinq defaites, **zero sortie d'ecran et zero bordure detruite**. Le bug de securite est corrige ; l'amelioration strategique n'est pas demontree. Les recompenses v3 et v4 etant differentes, leurs valeurs cumulees ne doivent pas etre comparees comme si elles mesuraient la meme chose. Les 23 tests passent.

Le dernier modele nouvellement entraine a aussi ete audite separement, sans modifier la selection sur validation : score moyen 35,4, cinq defaites, aucune sortie et aucune bordure detruite. Il collecte un peu plus, mais n'ameliore pas la recompense de validation ni le taux de victoire ; il est conserve pour analyse sans remplacer le checkpoint principal.

## Diagnostic des oscillations et progression par potentiel

La trace locale `runs/navigation_before_20261124/` montre que le PPO se deplace reellement, mais repete des cycles courts : 161 retours en deux decisions sur 305 decisions, notamment entre (542,630) et (554,630). Monter est autorise a cet endroit ; la politique choisit pourtant de revenir en arriere. La penalite de blocage ne se declenche pas puisque la position change.

L'experience suivante ajoute uniquement un wrapper de recompense, `PotentialShaping`, avec `F = gamma * Phi(suivant) - Phi(actuel)`. Le graphe sert a mesurer la distance aux objets restants ; il n'execute aucun trajet. Les entrees, les quatre actions, les poids de depart, la carte et les ghosts sont conserves.

Le potentiel est borne entre 0 et 22 : 20 fois la fraction d'objets deja collectes, plus 2/(1+distance en cases) vers un objet restant. La distance varie aussi a l'interieur d'une case, pour rendre visibles les petits aller-retours. Si aucun objet restant n'est accessible, une ressource citron peut servir de proxy de progression. Les murs interieurs sont relaxes dans cette estimation quand le casse-mur est actif ; la duree restante n'est pas simulee dans la recherche, ce qui en fait une heuristique optimiste.

Le gamma est celui du checkpoint (0,995). La valeur finale est nulle aux vraies terminaisons ; un timeout conserve le potentiel pour rester coherent avec le bootstrap. Les tests verifient l'identite de telescopage actualisee, l'absence de profit de shaping sur une boucle fermee, les objets inaccessibles, et l'invariance des observations et des transitions physiques.

La preservation theorique de l'objectif concerne les conditions du modele MDP et le retour actualise ; elle ne garantit ni une victoire ni une convergence rapide de notre reseau. La validation et la selection des checkpoints utilisent donc toujours la recompense de base, sans shaping, et le bilan compte separement collecte, boucles et victoires.

```powershell
.\.venv\Scripts\python.exe -m scripts.resume_ppo --model runs/ppo_20261124_020053/best_model.zip --timesteps 65536 --workers 2 --potential-shaping
```

Cette adaptation est une experience inspiree du potential-based reward shaping, pas une implementation de HRA, d'un GNN ou d'un agent A* qui piloterait Pac-Man.

Resultat du 24 novembre : reprise de 65 536 decisions dans `runs/resume_20261124_043056/`. La validation sans shaping passe de -122,052 a -105,70. Le checkpoint principal `runs/ppo_20261124_020053/best_model.zip` contient maintenant les nouveaux poids ; ceux de depart restent dans `runs/resume_20261124_043056/original_model.zip`.

Une verification de navigation sur seeds 9000 a 9004, distinctes de la validation de l'entrainement et des cinq parties de comparaison, confirme le changement : collecte moyenne de 17,4 % a 21,3 %, retours en deux decisions de 161 a 128,6 par partie (52,9 % a 36,3 % des decisions). Aucun incident de bordure.

Sur les cinq parties de test (1000, 7000 a 7003), le score moyen passe de 32 a 40,4. Sur la seed 1000, les retours en deux decisions passent de 161 a 132. Les cinq parties restent des defaites : l'oscillation est reduite, pas eliminee, et aucune victoire n'est demontree.

Le protocole et les metriques sont conserves dans `runs/resume_20261124_043056/navigation_selection/report.json`. Le choix du candidat utilise les seeds de validation 9000 a 9004, puis le test est execute sans changer la selection. Le filtre refuse une regression de collecte ou de boucles, sauf priorite a une augmentation des victoires, et conserve les sauvegardes. Les commandes de demonstration restent identiques.
