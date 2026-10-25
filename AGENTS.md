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

**Un resultat sur un athlete est une anecdote.** `cli cohort` rejoue
l'experience sur les athletes du cache et rend la DISTRIBUTION des resultats,
jamais leur moyenne — elle melangerait des historiques tres inegaux. Sur
soixante-et-un athletes, l'effet de charge mesure chez l'athlete de reference
se replique a p = 0,020 : 23 intervalles excluent zero en sa faveur, 33 ne
concluent pas, et 5 CONTRE — la charge y degrade la prevision. La domination de
la persistance se replique a p < 0,0001.

**Une conclusion peut etre juste et sous-dimensionnee.** A treize athletes ce
depot concluait que l'effet ne se generalisait pas (test du signe a 1,000) ; a
soixante-et-un il se replique. A quatorze athletes la dispersion inter-individus
sortait a 0,212 [0,000 · 0,393] et l'heterogeneite etait declaree non etablie ;
a soixante-deux elle vaut 0,205 [+0,131 · +0,278]. Les moyennes n'ont pas bouge,
les intervalles se sont resserres. Une conclusion negative s'enonce donc avec le
nombre de sujets qui la porte, et se relit quand ce nombre change.

**La mise en commun ne peut pas voir un athlete qui va contre la population.**
Le retrait vers la moyenne est ce qui identifie les constantes de temps, et
c'est aussi ce qui empeche l'ajustement hierarchique de produire un seul gain
negatif — pas meme chez les cinq athletes ou le protocole hors echantillon en
montre un. Les deux protocoles s'accordent en rang (correlation +0,406) et pas
en signe. Un verdict individuel se lit hors echantillon, jamais dans le
posterieur mis en commun.

**Identifie n'est pas valide.** Partagees sur soixante-deux historiques, tau1 et
tau2 sortent serrees (70,3 et 29,7 jours) et incompatibles avec la litterature
(42 et 12), tau2 poussant a 3,5 ecarts-types contre son prior. Un parametre qui
s'eloigne de son prior QUAND ON AJOUTE DES DONNEES signale un defaut de
specification, pas une decouverte. Le resserrement d'un posterieur ne vaut pas
validation.

**Un parametre se conteste en bougeant le protocole, pas en le commentant.**
L'hypothese etait que ces constantes apprennent la fenetre de lissage de
l'enveloppe. Mesure a fenetre 21 au lieu de 42 : tau1 perd 14,4 % et tau2
18,9 %, soit une elasticite de 0,22 et 0,30 — zero pour de la physiologie pure,
un pour un pur apprentissage de la fenetre. Les DEUX hypotheses sont donc
fausses, et environ un quart de ces constantes est du protocole. Trois pieges
sur ce chemin. `min_periods` s'applique a la fenetre, donc la raccourcir ecarte
des athletes : un troisieme ajustement de CONTROLE, a fenetre standard et
cohorte amputee des memes athletes, est ce qui disculpe la composition
(70,3 -> 71,7). La comparaison porte sur une cible qui a change : son
autocorrelation de rang 1 tombe de +0,539 a +0,343, `describe` la publie, et
sans ce chiffre on attribuerait a la physiologie une contraction de TOUS les
parametres de 8 a 19 %. Enfin un reglage non publie ecrit des artefacts
suffixes (`_f21`, `_n60`), parce qu'une experience qui met en doute une mesure
ne doit pas pouvoir l'ecraser.

**« Qui repond ? » se pose hors echantillon.** Chercher parmi douze
predicteurs le plus correle au gain puis le publier, c'est tester douze
hypotheses et n'en declarer qu'une. `cli responders` retire un athlete et
demande aux autres de le classer, pretraitement recalcule DANS chaque pli — le
calculer une fois sur tout l'echantillon ferait fuir la cible par le centrage.
Resultat : aucune famille ne depasse la classe majoritaire. Les deux nuls ne se
valent pas pour autant, et c'est le plancher qui les separe : la famille connue
AU PREMIER JOUR detecte un decalage de 0,05 sigma, donc son nul est bien
alimente et la question du demarrage a froid est FERMEE ; celle derivee de
l'historique n'atteint pas son plancher a 0,40 sigma, donc son nul n'est pas
etabli. Deux defauts sont sortis de cette mesure : un predicteur CONSTANT
(`sexe_feminin`, nul chez les 61 athletes) ressortait significatif parce que le
p permute d'un NaN vaut 0,0000, et six athletes declaraient un age NEGATIF qui
portait la correlation de l'age de +0,132 a +0,230. Un predicteur ecarte est
rapporte comme ecarte, et le prix de Bonferroni se paie sur les predicteurs
reellement testes.

**Un plan dit ce qu'il n'optimise pas.** Le planificateur cherche la charge
MINIMALE qui delivre le stimulus demande, et non la maximale que les
contraintes tolerent : sous une dose-reponse inconnue — elle existe chez
23 athletes sur 61, et nuisible chez 5 — le cout d'une charge inutile est reel la ou son
benefice ne l'est pas. `dose_response.py` mesure le verdict chez l'athlete
qu'on a en face et distingue « non detectee » d'« indeterminable », parce que
presenter une absence de mesure comme une absence d'effet est l'erreur que ce
depot s'emploie a ne pas commettre. Aucun des verdicts n'autorise a maximiser
une performance predite.

**Une borne de securite se mesure, et trois confonds l'attendent.**
`cli validate-plan` confronte les bornes du planificateur a 6768 semaines
reelles de 66 athletes : respect contre violation A L'INTERIEUR de chaque
athlete, puis test du signe. Le plancher de retention (ACWR >= 0,8) en sort
SOUTENU — +0,051 sigma, 37/55, p = 0,0145, detecte sans injection. Le plafond
d'ACWR et celui des jours consecutifs en sortent INDETERMINABLES, plancher
0,20 sigma. Trois pieges s'empilaient sur le chemin, et le protocole naif
annoncait que respecter le plafond d'ACWR COUTAIT 0,333 sigma a p < 0,0001 :
l'enveloppe, maximum glissant, a une variation mediane nulle a 28 jours ; un
maximum glissant qui regarde en arriere englobe la semaine qu'on juge, donc
l'effort dur de cette semaine entre dans sa propre issue ; et un maximum sur
plus de tirages est plus grand, les semaines a ACWR eleve etant suivies de
5,2 sorties de plus. Toute regle est donc jugee sur DEUX statistiques — le
meilleur effort et l'effort MEDIAN, insensible au nombre de tirages — et sur
DEUX regimes, dont un qui exige un entrainement regulier AVANT la semaine
jugee : un ACWR eleve est le plus souvent une reprise apres coupure, dont la
fenetre de reference est deprimee. Une regle qui ne tient que sur une des
quatre lignes ne tient pas.

**Un cas isole se juge contre un taux de base.** Cinq athletes sur
soixante-et-un voient la charge degrader leur prevision. Aucun artefact n'y a
ete etabli : trous de donnees, ruptures de niveau et ruptures d'echelle ont ete
compares a la DISTRIBUTION de la meme statistique chez les 23 detectes, et
quatre des cinq restent dans l'etendue. Sans groupe de comparaison, n'importe
quelle valeur a l'air suspecte. L'hypothese mecaniste la plus naturelle — une
charge qui tend fait extrapoler le modele vers le haut — est REFUTEE : le biais
median vaut -0,045 sigma, 27 athletes sur 61 sont surestimes, et
rho(biais, gain) = -0,018 sur 61 sujets. Ce qui gouverne le score est
l'amplitude de l'erreur, pas son signe.

**Un plan degrade ne se lit pas comme un plan optimal.** `optimize_week`
retombait silencieusement sur l'heuristique des qu'une contrainte rendait le
probleme infaisable. `WeeklyPlan.notes` porte desormais ce qui n'a pas ete
applique — plancher rabaisse, stimulus abandonne, borne aigu/chronique
inactive faute d'historique — et `xai_report` le publie. Une borne
silencieusement inactive est pire qu'une borne absente.

**Une charge doit disputer le signal au passage du temps.** Un premier
ajustement hierarchique laissait tau1 fuir a 127 jours, soit 4,6 ecarts-types
au-dela de son prior : a cette constante, la charge cumulee EST une tendance,
et le gain mesurait l'une en croyant mesurer l'autre. Tout modele de charge
porte donc un terme de tendance concurrent — il n'est pas la pour ameliorer
l'ajustement mais pour le lui disputer.

**Un seuil ne se recopie pas.** `readiness.py` annonce quand chaque capacite
devient disponible — 28 jours pour la borne aigu/chronique, trois mesures pour
l'enveloppe, huit plis pour le verdict. Chacun est celui qu'une fonction impose
vraiment, et les comptages passent par ces fonctions (`horizon_splits`,
`rolling_envelope`) plutot que de les reimplementer. Un seuil recopie derive le
jour ou la fonction change, et l'echelle devient un mensonge rassurant : un
test echoue si les deux divergent. Meme regle pour le prior de population, lu
dans l'artefact de cohorte et jamais ecrit en dur.

**Les tests n'ecrivent jamais dans `data/`.** Une fixture autouse le verifie
apres chaque test et echoue sinon. Le module de sensibilite ne deroge pas :
l'enveloppe perturbee n'existe qu'en memoire, et l'interface ne rend que des
statistiques — jamais une serie qu'on pourrait ecrire et confondre plus tard
avec une mesure.

## Organisation

```
README.md                presentation du projet
docs/resultats.md        journal complet des mesures
docs/methode.md          protocoles, limites, garde-fous
docs/architecture.md     modules, flux, mise en route

src/config.py            configuration unique, mise en cache
src/ingestion/           acces au jeu de donnees public (cache-first, md5)
src/layer_a_signal/      nettoyage, features, base de temps, performance
src/layer_b_twin/        modele de charge, modele bayesien,
                         mise en commun hierarchique des athletes
src/layer_c_planner/     contraintes de securite, optimiseur, parsing LLM,
                         verdict de dose-reponse par athlete
src/pipeline/            orchestrateur et provenance
src/evaluation/          metriques, references, validation, rapport,
                         modeles de fondation, sensibilite, cohorte, maturite,
                         confrontation des regles du plan aux donnees,
                         predicteurs de la reponse a la charge
app/                     tableau de bord Streamlit
```

## Commandes

```bash
pip install -e ".[dev]"
python -m src.cli ingest        # construire les artefacts
python -m src.cli calibrate     # reporter les estimations dans params.yaml
python -m src.cli evaluate      # cinq protocoles, planchers, provenance
python -m src.cli cohort        # replication sur le cache d'athletes
python -m src.cli fetch-athletes --critere velo|course   # elargir le cache
python -m src.cli plan          # semaine proposee, et ce qu'elle ne promet pas
python -m src.cli validate-plan # ce que les bornes du plan valent face aux donnees
python -m src.cli responders    # qui repond, teste hors echantillon
python -m src.cli pool          # ajustement hierarchique (lent : des heures)
python -m src.cli pool --decrire               # la cible seule, en minutes
python -m src.cli pool --fenetre 21            # contester les constantes de temps
python -m src.cli pool --exclure UUID,UUID     # controler la composition de la cohorte
pytest                          # hors ligne par defaut

pip install -e ".[tsfm]"        # torch + Chronos-2 + TTM, optionnels
python -m src.cli evaluate --models chronos2,ttm
ruff check src app tests
streamlit run app/main_dashboard.py
```

## Documentation

`README.md` est la presentation du projet, pour un lecteur qui le decouvre :
concis, et aucun chiffre qui ne soit reproductible par une commande. Le
journal complet des mesures vit dans `docs/resultats.md`, les protocoles et
garde-fous dans `docs/methode.md`, l'architecture dans
`docs/architecture.md`. Un resultat nouveau s'ecrit dans `docs/`, et ne
remonte au README que s'il change ce que le projet conclut.

## Style

Commentaires et docstrings en francais, en expliquant **pourquoi** plutot que
**quoi**. Un commentaire qui paraphrase le code suivant ne sert a rien ; un
commentaire qui explique pourquoi un seuil vaut 42 plutot que 30, ou pourquoi
une garde existe, evite qu'on le defasse six mois plus tard.

Messages de commit a l'imperatif, corps explicatif quand le changement n'est
pas evident. Pas de sections normalisees ni d'emojis de statut.
