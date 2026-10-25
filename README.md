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

Et le constat se réplique largement : sur **61 athlètes**, trois seulement
battent la persistance, test du signe à **p < 0,0001**.

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
**hétérogène** : présent chez **23 athlètes sur 61**, absent chez 33 — et
franchement **nuisible chez 5**, dont l'intervalle exclut zéro dans l'autre
sens. Aucune variable disponible à l'inscription ne distingue les groupes.

Le résultat nul n'est donc pas annulé — il est circonscrit.

---

## Le fonctionnement

Huit commandes, qui produisent tous les chiffres ci-dessus.

```bash
python -m src.cli ingest         # 917 séances → charge, performance, provenance
python -m src.cli calibrate      # FC max/repos estimées, jamais supposées
python -m src.cli evaluate       # 5 protocoles + planchers de détectabilité
python -m src.cli cohort         # réplication sur 61 athlètes
python -m src.cli plan           # une semaine, et ce qu'elle ne promet pas
python -m src.cli validate-plan  # ce que ses bornes valent face aux données
python -m src.cli responders     # qui répond, testé hors échantillon
python -m src.cli pool           # modèle hiérarchique sur la cohorte (~2 h 30)
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

L'athlète de référence est l'un des 23 répondeurs sur 61. Chez les 38 autres,
la dernière phrase devient « ce plan n'optimise aucune performance prédite, et
ne peut pas en promettre » — et c'est l'information utile.

Ses bornes de sécurité, elles, viennent de la littérature, et c'est une raison
de les écrire plutôt qu'une mesure. `validate-plan` les confronte à
6768 semaines réelles de 66 athlètes : le plancher de retention en sort
**soutenu** (+0,051 σ sur l'effort médian, 37 athlètes sur 55, p = 0,0145), le
plafond de charge aiguë et celui des jours consécutifs **indéterminables** à
0,20 σ près. Une borne indéterminable reste appliquée — mais le dépôt dit
maintenant laquelle il a mesurée.

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
venait de l'athlète de référence du dépôt. Il s'est avéré être l'un des 23
répondeurs sur 61 — réel, mais pas typique.

**Ajuster n'est pas prédire, et j'ai pu le chiffrer.** Un modèle hiérarchique
sur 62 athlètes trouve un effet de charge excluant zéro chez **61 d'entre
eux**. Le protocole hors échantillon, sur les mêmes données, n'en trouve que
**23 sur 61**. Même modèle, mêmes athlètes : il explique le passé chez tous
sauf un et n'en prévoit la suite que chez un peu plus d'un tiers. C'est la
signature du surapprentissage, et c'est pourquoi aucun ajustement n'est publié
ici sans son pendant hors échantillon.

**Une conclusion peut être juste et sous-dimensionnée.** À quatorze athlètes,
la dispersion de l'effet entre individus sortait à 0,21 avec un intervalle
touchant zéro, et j'ai écrit que l'hétérogénéité n'était pas établie. À 62
athlètes elle vaut 0,205 [+0,131 · +0,278] : la moyenne n'a pas bougé, seul
l'intervalle s'est resserré. Je ne m'étais pas trompé sur le nombre, mais sur
ce qu'il permettait d'affirmer — et c'est précisément ce qu'un plancher de
détectabilité sert à anticiper.

**La mise en commun a rendu ce qu'on lui demandait, et posé un autre problème.**
Les constantes de temps du modèle de Banister ne sont pas identifiables sur un
athlète — le postérieur ne resserre τ₁ que de 41 % par rapport à son prior.
Partagées entre 62 athlètes, elles sortent serrées : 70,3 jours [52,0 · 88,4]
et 29,7 [23,3 · 37,0]. Sauf que la littérature dit 42 et 12, et que la seconde
pousse à 3,5 écarts-types contre son prior. Un paramètre qui s'éloigne de son
prior **quand on ajoute des données** ne se lit pas comme une mesure
physiologique. Identifiées n'est pas validé, et seule la première moitié était
l'objectif.

**Un quart de ces constantes est du protocole, et je ne sais pas d'où vient le
reste.** L'enveloppe étant un maximum glissant sur 42 jours, l'hypothèse était
que le modèle apprend la fenêtre de lissage. Réajusté à fenêtre 21, τ₁ perd
14,4 % et τ₂ 18,9 % — soit une élasticité de 0,22 et 0,30, là où de la
physiologie pure donnerait 0 et un pur apprentissage de la fenêtre donnerait 1.
Les deux hypothèses sont donc fausses. Un troisième ajustement de contrôle
était nécessaire pour l'affirmer : raccourcir la fenêtre écarte deux athlètes
faute d'observations, et un τ partagé aurait bougé pour deux raisons mêlées. À
fenêtre constante, retirer ces deux athlètes ne déplace rien (70,3 → 71,7).
Ce qui bouge, en revanche, c'est toute la ligne de paramètres, de 8 à 19 % —
la cible est simplement moins régulière à 21 jours, son autocorrélation
tombant de +0,539 à +0,343. Ce qui survit : σ net exclut zéro dans les trois
ajustements, et les gains individuels corrèlent à +0,839.

**Deux lignes de code auraient suffi à inventer un résultat.** En cherchant ce
qui distingue un répondeur, le seul prédicteur que ma première exécution a
déclaré significatif était `sexe_feminin` — qui vaut 0 chez les soixante-et-un
athlètes du cache. Une colonne constante n'a pas de corrélation définie, et le
p permuté d'un NaN ressort à 0,0000. Dans la même passe, six athlètes
déclaraient une année de naissance postérieure à la publication du jeu de
données, donc des âges de −58 à −42 ans, ce qui portait la corrélation de
l'âge de +0,132 à +0,230 et en faisait la deuxième du tableau. Ni l'un ni
l'autre n'aurait levé une erreur. Un prédicteur écarté est maintenant rapporté
comme écarté : un tableau qui ne dit pas ce qu'il a retiré se lit comme si
tout avait été testé.

**Un résultat massif peut être entièrement un artefact.** En confrontant les
bornes du planificateur aux données, mon premier protocole a répondu que
respecter le plafond de charge aiguë **coûtait 0,333 σ de performance**, avec
7 athlètes sur 55 en faveur et p < 0,0001. J'étais à un commit de démonter une
règle de sécurité. Trois confonds s'empilaient : une cible qui ne bouge que
par paliers, la semaine évaluée qui entrait dans sa propre issue, et un maximum
mécaniquement plus grand quand on l'évalue sur plus de sorties. Les corriger ne
laisse rien : −0,042 σ, p = 0,18. Un p très petit mesure la cohérence d'un
protocole avec lui-même, pas sa validité.

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

**« Qui répond ? » est à moitié fermée, et la mauvaise moitié reste ouverte.**
Posée hors échantillon — on retire un athlète et on demande aux soixante
autres de dire s'il répond — la réponse est non : aucune famille de
prédicteurs ne dépasse « prédire la classe majoritaire ». Pour ce qu'on sait
le **premier jour**, le test est sensible (plancher 0,05 σ) et le nul tient :
rien, avant la première séance, ne distingue un répondeur. Pour les
prédicteurs **dérivés de l'historique**, le plancher n'est pas atteint même à
0,40 σ — onze variables sur 61 sujets diluent tout. Celui-là n'est pas un
résultat, c'est un manque de sujets. La voie n'est donc pas plus de
prédicteurs, mais plus d'athlètes, ou un prédicteur postulé à l'avance plutôt
que cherché parmi douze.

**Les cinq cas négatifs sont instruits, et ils restent ouverts.** Aucun
artefact établi — les trous de données, ruptures de niveau et ruptures
d'échelle de ces cinq athlètes restent dans l'étendue des 23 répondeurs. Mon
hypothèse mécaniste (une charge qui tend fait extrapoler le modèle vers le
haut) est réfutée : le biais est symétrique sur les 61. Les deux écarts qui
subsistent — pente de charge et amplitude d'erreur — tombent à p = 0,074 et
0,070, juste sous ce que cinq sujets permettent de voir.

**Trouver d'où vient le reste des constantes de temps.** La fenêtre de lissage
en explique environ un quart, c'est mesuré ; les trois quarts restants ne le
sont pas, et l'écart à la littérature ne se referme pas en corrigeant la
fenêtre. Les pistes qui restent à départager : le pas de 28 jours entre deux
points d'enveloppe, le `min_periods` qui conditionne ce qui entre dans la
cible, et la forme même du noyau de convolution.

**Reformuler la contrainte de jours consécutifs.** La mesure ne conclut pas sur
le plafond à 3 jours, mais sa ventilation est claire : l'issue monte de 0 à 3
jours consécutifs puis retombe à partir de 4. Le seuil est posé au bon endroit
et la contrainte « ≤ 3 » dilue le gain qu'elle devrait capter, puisqu'elle
accepte aussi 0, 1 et 2. C'est un défaut de formulation, et il est corrigeable.

**Tester les plafonds par type de séance.** `validate-plan` ne les atteint pas :
les résumés publics ne portent pas les types. Un export personnel les
porterait, et c'est la moitié du planificateur qui reste non mesurée.

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
