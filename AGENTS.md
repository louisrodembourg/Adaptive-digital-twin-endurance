# Conventions de contribution

## Ce que fait ce projet

Un jumeau numerique d'entrainement d'endurance, construit sur un jeu de
donnees public (GoldenCheetah OpenData). Il ingere des seances, en derive une
charge d'entrainement et une mesure de performance, ajuste un modele
fitness-fatigue, puis **evalue honnetement si ce modele predit quoi que ce
soit**.

Le dernier point est le coeur du projet, pas une formalite. La reponse
mesuree est que le modele de Banister n'apporte rien au-dela d'une simple
persistance sur ces donnees, et le depot est organise pour rendre ce constat
verifiable plutot que pour le masquer.

## Regles non negociables

**Aucune donnee inventee ne doit pouvoir passer pour mesuree.** Toute seance
porte une `DataSource` explicite ; `add_session` n'a pas de valeur par
defaut ; ecrire du synthetique exige `allow_synthetic=True`. Les generateurs
de donnees vivent dans `tests/fixtures/`, jamais dans `src/`. Cette regle
existe parce que le tableau de bord a longtemps affiche 84 seances generees
sous l'etiquette `fit_file`.

**Aucune metrique ne compte des lignes.** Les series ne sont pas garanties a
1 Hz sans trou. Tout passe par `src/layer_a_signal/timebase.py` et
s'exprime en secondes reelles.

**Aucune constante physiologique dans le code.** Elles vivent dans
`config/params.yaml` et sont lues via `src/config.py`. Les valeurs propres a
un athlete sont `null` tant que `cli calibrate` n'a pas tourne : une erreur
explicite vaut mieux qu'un nombre fonde sur une supposition.

**Aucune amelioration n'est affirmee sans validation hors echantillon.** Un
R2 d'ajustement ne prouve rien. Toute variante passe par
`src/evaluation/`, contre les references declarees dans `baselines.py`. Les
decoupages respectent l'ordre du temps.

**Ni sans intervalle quand l'ecart est mince.** Deux variantes evaluees sous
le meme decoupage predisent les memes points : soustraire leurs R2 ne dit pas
si l'ecart depasse le bruit. `report.paired_table` fait un bootstrap apparie
par blocs — par plis, pas par jours, les erreurs d'un pli etant autocorrelees.
Chronos-2 arrive premier de 0,003 de R2 sur la persistance ; l'intervalle dit
que ce n'est rien, et c'est l'intervalle qui est publie.

**Une variante qui n'a rien execute ne publie rien.** `cross_validate` avale
les exceptions d'un prediteur : une variante cassee ressort en `folds = 0`
avec des NaN partout, ce qui ressemble a un resultat. `cli evaluate` sort en
erreur dans ce cas, et avertit quand deux variantes n'ont pas couvert le meme
nombre de plis — un modele dont la tete est entrainee sur un horizon court
echoue sur les plis longs et sa ligne porte alors sur moins de points.

**Aucune capacite annoncee n'est crue sur parole.** La fiche de TTM annonce
l'integration de variables exogenes ; en zero-shot son dorsal est independant
par canal et la prevision ne bouge pas d'un bit quand on remplace les
covariables. Un test le mesure, et `SPECS` dans `foundation.py` porte ce que
chaque modele fait reellement — pas ce que sa fiche promet.

**Une charge ne s'oppose qu'a une cible de sa discipline.** Le resultat nul le
plus cite du depot opposait une charge aux deux tiers cycliste a une vitesse
en course a pied. Apparie correctement — charge velo contre puissance velo —
le modele de charge bat « predire la moyenne » de +0,339 avec un intervalle
[+0,055 · +0,524]. Les variantes portent leur discipline dans leur nom, et la
charge de course sert de controle negatif sur la cible velo : si elle y
predisait quelque chose, c'est une fuite qu'il faudrait chercher, pas un
transfert physiologique.

**Un resultat nul s'accompagne de son plancher.** Affirmer l'absence d'effet
sans dire quel effet le protocole saurait voir, c'est confondre « on n'a rien
trouve » avec « il n'y a rien ». `sensitivity.py` injecte dans l'enveloppe
reelle un effet d'amplitude connue, de la forme que le modele postule, et rend
le plus petit que l'intervalle apparie distingue du hasard : 2 sigma contre la
persistance sur la cible course, 1 sigma sur la cible velo. Ce plancher est
OPTIMISTE, le signal injecte ayant exactement la forme ajustee.

**Un maximum ne se prend pas deux fois.** `perf_index` prenait un maximum sur
les durees avant que l'enveloppe en reprenne un sur 42 jours, d'ou
l'autocorrelation a 0,97 qui rend la persistance imbattable par construction.
Les durees sont publiees une par une, et une duree comptant moins de quarante
observations est ecartee ET rapportee comme ecartee.

**Les tests n'ecrivent jamais dans `data/`.** Une fixture autouse le verifie
apres chaque test et echoue sinon. Le module de sensibilite ne deroge pas :
l'enveloppe perturbee n'existe qu'en memoire, et l'interface ne rend que des
statistiques — jamais une serie qu'on pourrait ecrire et confondre plus tard
avec une mesure.

## Organisation

```
src/config.py            configuration unique, mise en cache
src/ingestion/           acces au jeu de donnees public (cache-first, md5)
src/layer_a_signal/      nettoyage, features, base de temps, performance
src/layer_b_twin/        modele de charge et modele bayesien
src/layer_c_planner/     contraintes de securite, optimiseur, parsing LLM
src/pipeline/            orchestrateur et provenance
src/evaluation/          metriques, references, validation, rapport,
                         modeles de fondation, sensibilite
app/                     tableau de bord Streamlit
```

## Commandes

```bash
pip install -e ".[dev]"
python -m src.cli ingest        # construire les artefacts
python -m src.cli calibrate     # reporter les estimations dans params.yaml
python -m src.cli evaluate      # cinq protocoles, planchers, provenance
pytest                          # hors ligne par defaut

pip install -e ".[tsfm]"        # torch + Chronos-2 + TTM, optionnels
python -m src.cli evaluate --models chronos2,ttm
ruff check src app tests
streamlit run app/main_dashboard.py
```

## Style

Commentaires et docstrings en francais, en expliquant **pourquoi** plutot que
**quoi**. Un commentaire qui paraphrase le code suivant ne sert a rien ; un
commentaire qui explique pourquoi un seuil vaut 42 plutot que 30, ou pourquoi
une garde existe, evite qu'on le defasse six mois plus tard.

Messages de commit a l'imperatif, corps explicatif quand le changement n'est
pas evident. Pas de sections normalisees ni d'emojis de statut.
