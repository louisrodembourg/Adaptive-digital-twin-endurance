# Digital Twin — Endurance

Un jumeau numérique d'entraînement : il ingère des séances réelles, en dérive
une charge et une mesure de performance, ajuste un modèle physiologique — puis
vérifie si ce modèle prédit quoi que ce soit.

**Il conclut que non.** C'est le résultat principal du projet, et il se
reproduit en une commande.

---

## Pourquoi

Le modèle fitness-fatigue de Banister est enseigné depuis 1975 et implémenté
dans la plupart des plateformes d'entraînement. Il est rarement validé hors
échantillon.

Je voulais savoir ce qu'il valait sur des données réelles. La question s'est
vite déplacée : construire un modèle est facile, construire l'appareil de
mesure qui dit honnêtement s'il marche est le vrai travail. Ce dépôt est
surtout cet appareil de mesure.

---

## Le concept

Trois couches, et une règle.

```
séances brutes  →  charge + performance  →  modèle  →  plan
   couche A            couche A            couche B   couche C
```

La règle : **aucune amélioration n'est affirmée sans validation hors
échantillon, ni sans intervalle quand l'écart est mince.** Toute variante est
confrontée à des références déclarées d'avance — dont « répéter la dernière
valeur », qui se révèle difficile à battre. Un R² d'ajustement seul n'est
jamais publié.

---

## Le résultat

Sur 917 séances, en validation croisée temporelle :

| variante | R² ajustement | R² hors échantillon |
|---|---:|---:|
| **persistance** (répéter la dernière valeur) | −0,023 | **+0,318** |
| moyenne mobile exponentielle | −0,215 | +0,081 |
| prédire la moyenne | 0,000 | −0,014 |
| **Banister (fitness + fatigue)** | **+0,107** | **−0,090** |

Banister a le **meilleur ajustement et le pire score hors échantillon** : il
mémorise, il n'apprend pas. Le postérieur bayésien dit la même chose par une
autre voie — effet net de la charge **−0,000, intervalle [−0,020 · 0,017]**,
sur un échantillonnage convergé. Un modèle de fondation (Chronos-2, 120 M de
paramètres) nourri de la charge passée *et future* n'extrait rien de plus que
l'inertie.

Et le constat se réplique : sur treize athlètes, un seul bat la persistance,
test du signe à **p = 0,003**.

### Un seul effet a survécu à son intervalle

En cherchant pourquoi, j'ai trouvé une erreur de conception : la cible mesurait
une vitesse **en course à pied** alors que 68 % de la charge opposée était du
**vélo**. Apparié correctement — charge vélo contre puissance vélo, sur 417
observations jamais exploitées jusque-là :

| comparaison | gain | IC 95 % |
|---|---:|---|
| charge vélo **contre charge course** | +0,631 | [+0,407 · +0,742] |
| charge vélo **contre prédire la moyenne** | +0,339 | [+0,055 · +0,524] |

Apparier la discipline vaut +0,63 de compétence. Mais l'effet est
**hétérogène** : présent chez quatre athlètes sur treize, absent chez les neuf
autres, et aucune variable disponible à l'inscription ne les distingue.

Le résultat nul n'est donc pas annulé — il est circonscrit.

---

## Le fonctionnement

Cinq commandes, qui produisent tous les chiffres ci-dessus.

```bash
python -m src.cli ingest      # 917 séances → charge, performance, provenance
python -m src.cli calibrate   # FC max/repos estimées, jamais supposées
python -m src.cli evaluate    # 5 protocoles + planchers de détectabilité
python -m src.cli cohort      # réplication sur 13 athlètes
python -m src.cli plan        # une semaine, et ce qu'elle ne promet pas
python -m src.cli pool        # modèle hiérarchique sur la cohorte (~25 min)
```

Le planificateur mérite un mot, parce qu'il découle directement de la mesure.
Il cherchait à **maximiser** la charge sous un plafond de sécurité — autrement
dit « entraîne-toi autant que tu peux ». Puisque la dose-réponse est absente
chez deux athlètes sur trois, le coût d'une charge inutile est réel là où son
bénéfice ne l'est pas. Il cherche donc la charge **minimale** qui délivre le
stimulus demandé, et il affiche le verdict mesuré chez l'athlète qu'il a en
face :

```
$ python -m src.cli plan

Ce plan cherche la charge la PLUS FAIBLE qui délivre le stimulus demandé,
sous les contraintes de sécurité.
Dose-reponse detectee chez cet athlete : competence +0.339 contre « moyenne »,
IC 95 % [+0.055 · +0.524], 38 plis.
Un effet de charge est donc démontré chez cet athlète — ce plan ne le maximise
pas pour autant, il délivre un stimulus au moindre coût.
```

L'athlète de référence est l'un des quatre répondeurs. Chez les neuf autres, la
dernière phrase devient « ce plan n'optimise aucune performance prédite, et ne
peut pas en promettre » — et c'est l'information utile.

Ce verdict demande plus de quatre cents jours d'historique. `cli state` rend
donc une **échelle de maturité** : quelles capacités sont disponibles
aujourd'hui, ce qui manque aux autres en jours, et — tant que l'athlète n'a pas
son propre verdict — ce que la cohorte permet d'affirmer en attendant. Détail
dans [la méthode](docs/methode.md#léchelle-de-maturité).

---

## La technologie

`pandas` · `numpy` · `scipy` · `PyMC` / `ArviZ` · `OR-Tools` CP-SAT ·
`Streamlit` · `pytest` · `ruff`

En option : `torch` · `chronos-forecasting` · `granite-tsfm`, pour les modèles
de fondation de séries temporelles.

Trois choix qui ont compté :

- **PyMC plutôt qu'un ajustement ponctuel.** C'est le postérieur qui a révélé
  que k₁ et k₂ sont corrélés à 0,986 et que le modèle n'est pas identifié. Un
  `curve_fit` aurait rendu k₁ = 0,050 et personne n'aurait vu le problème.
- **OR-Tools CP-SAT pour la planification.** Les contraintes d'entraînement
  sont combinatoires — espacement dur/facile, bornes de répétition, rampe de
  charge — pas continues.
- **Chronos-2 comme arbitre non paramétrique.** Il accepte des covariables
  futures connues, donc il répond à « la charge porte-t-elle de l'information
  sur la performance ? » sans supposer la forme de la relation.

286 tests, hors ligne par défaut. Une fixture vérifie après chaque test que
rien n'a été écrit dans `data/`.

---

## Les réflexions

Ce que ce projet m'a appris, dans l'ordre d'importance.

**L'estimande compte plus que le modèle.** La performance était mesurée par le
meilleur effort de chaque séance — autocorrélation −0,12, du bruit autour d'un
niveau. Aucun modèle ne pouvait en tirer quoi que ce soit. Changer la question
posée — « quel est le meilleur dont cet athlète ait été capable récemment ? » —
a fait plus que n'importe quel changement d'algorithme.

**Un écart sans intervalle ne vaut rien.** Chronos-2 arrive premier du tableau,
de 0,003 de R² sur la persistance. Le bootstrap apparié rend
[−0,172 · +0,255] : ce n'est rien. C'est l'intervalle qui est publié.

**« On n'a rien trouvé » n'est pas « il n'y a rien ».** J'ai injecté un effet
synthétique d'amplitude connue dans les données réelles pour mesurer ce que le
protocole saurait voir. Il faudrait un effet **deux fois plus dispersé que
l'enveloppe entière** pour battre la persistance. Le constat tient, mais la
comparaison qui le porte n'a presque aucune puissance — et c'est désormais
chiffré au lieu d'être supposé.

**Une capacité annoncée n'est pas une capacité.** La fiche d'un des modèles de
fondation annonce l'intégration de variables exogènes. En remplaçant
entièrement les covariables, sa prévision est identique **au bit près**. Un
test le mesure.

**Un résultat sur un athlète est une anecdote.** Mon premier effet positif
venait de l'athlète de référence du dépôt. Il s'est avéré être l'un des quatre
répondeurs sur treize — réel, mais pas typique.

**Ajuster n'est pas prédire, et j'ai pu le chiffrer.** Un modèle hiérarchique
sur quatorze athlètes trouve un effet de charge excluant zéro chez **12 d'entre
eux**. Le protocole hors échantillon, sur les mêmes données, n'en trouve que
**4 sur 13**. Même modèle, mêmes athlètes : il explique le passé chez presque
tous et n'en prévoit la suite que chez un tiers. C'est la signature du
surapprentissage, et c'est pourquoi aucun ajustement n'est publié ici sans son
pendant hors échantillon.

**La mise en commun a quand même rendu quelque chose.** Les constantes de temps
du modèle de Banister ne sont pas identifiables sur un athlète — le postérieur
ne resserre τ₁ que de 41 % par rapport à son prior. Partagées entre quatorze
athlètes, elles sortent à 51,7 jours [29,2 · 76,3] et 19,0 [9,9 · 27,9]. Le
gain ne dépendait d'aucun résultat favorable sur la charge, juste du nombre de
sujets.

**Le code mort mentait.** Trois règles de sécurité existaient sans être
branchées : le rapport charge aiguë sur charge chronique, un seuil de VFC lu
sous une clé absente de la configuration, et une indisponibilité déclarée par
l'athlète posée en simple suggestion au solveur. Chacune donnait l'apparence
d'une garantie.

**Une note d'ingénierie, aussi.** Le premier ajustement MCMC demandait
22 heures : avec des paramètres corrélés à 0,99, l'échantillonneur ne parcourt
la crête qu'à tout petits pas. Échantillonner l'effet net plutôt que chaque
gain — même modèle, autre paramétrisation — le ramène à **26 minutes**.

---

## Ce que je ferais ensuite

Élargir la cohorte au-delà de quatorze sujets. C'est devenu le facteur
limitant : le modèle hiérarchique estime la dispersion entre athlètes à
0,21 avec un intervalle qui touche zéro, donc sur ce nombre de sujets il ne
peut pas démontrer qu'ils diffèrent. Et « qui répond ? » reste sans réponse,
pour la même raison.

---

## Pour aller plus loin

| | |
|---|---|
| [**Résultats**](docs/resultats.md) | le journal complet des mesures, tous les tableaux |
| [**Méthode**](docs/methode.md) | protocoles, ce qui est réel, limites connues, garde-fous |
| [**Architecture**](docs/architecture.md) | modules, flux de données, mise en route |
| [**AGENTS.md**](AGENTS.md) | les règles non négociables du dépôt |

Données : [GoldenCheetah OpenData](https://osf.io/6hfpz/) — public, anonymisé,
sans coordonnées GPS. Licence du dépôt : MIT.
