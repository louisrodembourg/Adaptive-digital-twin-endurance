# Résultats mesurés

Le journal complet des mesures, dans l'ordre où elles ont été faites. Chaque
chiffre se reproduit par `python -m src.cli evaluate` ou `cohort` sur un jeu
de données public — c'est ce qui les rend vérifiables plutôt que déclaratifs.

Présentation courte du projet : [README](../README.md).
Protocoles et garde-fous : [méthode](methode.md).

---

## Le résultat

Sur 917 séances réelles, en validation croisée temporelle :

| variante | R² ajustement | R² hors échantillon |
|---|---:|---:|
| **persistance** (répéter la dernière valeur) | −0,023 | **+0,318** |
| moyenne mobile exponentielle | −0,215 | +0,081 |
| tendance temporelle | +0,271 | +0,042 |
| prédire la moyenne | 0,000 | −0,014 |
| fitness seul | +0,066 | −0,060 |
| **Banister (fitness + fatigue)** | **+0,107** | **−0,090** |

Trois choses s'y lisent.

**Banister a le meilleur ajustement et le pire score hors échantillon.** Il
mémorise, il n'apprend pas. C'est la signature même du surapprentissage,
et elle réplique la critique publiée dans [*Scientific Reports* (2025)](https://www.nature.com/articles/s41598-025-88153-7)
sur le mauvais conditionnement du modèle fitness-fatigue.

**La tendance passe de +0,271 à +0,042.** Un bon ajustement ne vaut rien en
soi — c'est pour cela que ce dépôt ne publie jamais un R² d'ajustement seul.

**Ce qui prédit, c'est l'inertie.** La persistance obtient +0,318, et aucun
modèle de charge ne la bat. C'est le constat le plus robuste du dépôt : il se
réplique sur treize athlètes, un seul au-dessus, test du signe à p = 0,003.

Cette dernière phrase demande une précision que le dépôt a longtemps omise :
battre la persistance sur une cible autocorrélée à 0,97 exigerait un effet de
charge **deux fois plus dispersé que l'enveloppe entière**. Le constat tient,
mais la comparaison qui le porte n'a presque aucune puissance, et c'est
mesuré — voir [le plancher de détectabilité](#le-plancher-de-détectabilité-et-ce-quil-fait-à-la-conclusion).

---

## Ce qui rend ce résultat possible

Le projet mesurait initialement la performance par le **meilleur effort de
chaque séance**. C'est un mauvais estimande : il décrit ce qu'*était* la
séance — fractionné ou récupération — pas la capacité de l'athlète.

| observable | autocorrélation (retard 1) |
|---|---:|
| meilleur effort de séance, course | −0,12 |
| meilleur effort de séance, puissance 20 min | −0,00 |
| **enveloppe glissante 42 j, course** | **+0,79** |
| **enveloppe glissante 42 j, vélo** | **+0,93** |

Du bruit autour d'un niveau : aucun modèle ne pouvait en tirer quoi que ce
soit. L'enveloppe répond à une autre question — « quel est le meilleur dont
cet athlète ait été capable récemment ? » — et devient prédictible.

Contrepartie assumée : une enveloppe est autocorrélée **par construction**,
donc facile à prédire par simple inertie. C'est exactement pourquoi la
persistance figure parmi les références obligatoires ; sans elle, le +0,318
se raconterait comme un succès du modèle.

---

## Ce que dit le postérieur

Le modèle bayésien est ajusté par MCMC sur les 1 987 jours de charge et les
85 points d'enveloppe. L'échantillonnage converge proprement — `r_hat` à
1,0000 sur tous les paramètres, aucune divergence, ESS minimal de 746 — ce
qui rend ses diagnostics interprétables.

| paramètre | moyenne | écart-type | intervalle 94 % |
|---|---:|---:|---|
| τ₁ (fitness, jours) | 29,3 | 11,7 | [9,1 · 51,1] |
| τ₂ (fatigue, jours) | 13,6 | 4,8 | [4,5 · 22,1] |
| k₁ (gain fitness) | 0,050 | 0,032 | [0,007 · 0,112] |
| k₂ (gain fatigue) | 0,050 | 0,040 | [−0,004 · 0,127] |
| **k₁ − k₂ (effet net)** | **−0,000** | **0,010** | **[−0,020 · 0,017]** |

**L'effet net de la charge d'entraînement sur la performance est
indiscernable de zéro.** Ce n'est pas une estimation imprécise faute de
données : l'intervalle est étroit, centré sur zéro, et issu d'un
échantillonnage convergé.

**k₁ et k₂ sont corrélés à 0,986.** Les données ne distinguent pas un gain de
fitness élevé compensé par une pénalité de fatigue élevée d'une paire de
gains modestes — seule leur différence est contrainte. Le postérieur de τ₁ ne
se resserre que de 41 % par rapport à son prior : les constantes de temps ne
sont pas identifiées non plus.

**L'intervalle de prédiction à 95 % ne couvre que 34 % des observations.** Le
modèle est massivement surconfiant.

Un postérieur plat est un résultat, pas un échec. C'est le seul usage honnête
de l'inférence bayésienne ici : montrer ce que les données contraignent
réellement, là où un ajustement ponctuel aurait présenté k₁ = 0,050 comme une
valeur estimée.

> **Note d'ingénierie.** Cette géométrie coûte cher : avec des paramètres
> corrélés à 0,99, NUTS ne parcourt la crête qu'à tout petits pas, et le
> premier ajustement a demandé **22 heures**. Échantillonner l'effet net
> plutôt que chaque gain — même modèle, autre paramétrisation — le ramène à
> **26 minutes**, soit un facteur 51.

---

## Ce que disent les modèles de fondation

Le résultat ci-dessus repose sur **un seul modèle paramétrique**. Une
objection reste donc ouverte : peut-être que la charge prédit la performance,
mais pas sous la forme d'une convolution exponentielle à deux constantes de
temps.

Les modèles de fondation de séries temporelles permettent de la tester sans
supposer de forme fonctionnelle. [Chronos-2](https://huggingface.co/amazon/chronos-2)
accepte des covariables passées et futures connues : on lui donne l'enveloppe
comme cible et la charge comme covariable, puis on le compare **au même
modèle privé de la covariable**. L'écart entre les deux lignes est une mesure
non paramétrique de l'information que la charge porte sur la performance.

Les covariables fournies ne sont pas le seul TRIMP brut : elles incluent
`fitness` et `fatigue`, calculés par la même fonction que celle qui alimente
le modèle bayésien. Le modèle reçoit donc **exactement les régresseurs de
Banister**, libre de les combiner autrement. Un résultat nul ne pourra pas
s'expliquer par une mise en forme insuffisante.

### Le résultat

Protocole à horizon calendaire de 28 jours, 18 plis à origine glissante,
545 jours d'enveloppe définie :

| variante | R² ajustement | R² hors échantillon | plis |
|---|---:|---:|---:|
| **Chronos-2 + charge** | — | **+0,657** | 18 |
| **persistance** | −0,008 | **+0,654** | 18 |
| Chronos-2 sans charge | — | +0,621 | 18 |
| TTM sans charge | — | +0,514 | 17 |
| moyenne mobile exponentielle | −0,001 | +0,503 | 18 |
| tendance temporelle | +0,281 | +0,122 | 18 |
| prédire la moyenne | +0,000 | +0,090 | 18 |
| fitness seul (τ 60) | +0,099 | +0,050 | 18 |
| Banister (τ 42/12) | +0,100 | −0,017 | 18 |

Chronos-2 nourri de la charge arrive premier. **De +0,003.** Un classement
n'est pas une mesure, et un écart de cette taille ne veut rien dire tant
qu'on ne l'a pas borné. Les deux variantes prédisent exactement les mêmes
points sous le même découpage, ce qui autorise un bootstrap apparié — par
**blocs**, puisque les erreurs d'un même pli sont fortement autocorrélées et
que tirer des jours indépendamment supposerait 545 observations
indépendantes là où il y en a 18 :

| comparaison | gain | intervalle 95 % | P(gain) |
|---|---:|---|---:|
| Chronos-2 + charge **contre** Chronos-2 sans charge | +0,096 | [−0,121 · +0,430] | 0,70 |
| Chronos-2 + charge **contre** persistance | +0,008 | [−0,172 · +0,255] | 0,55 |

**Aucun des deux écarts ne survit.** Le gain apporté par la charge est
compatible avec zéro, et le premier rang de Chronos-2 face à la persistance
est un tirage à pile ou face. Un modèle de 120 millions de paramètres,
entraîné sur des corpus de séries que ce projet n'a pas, à qui l'on donne la
charge passée *et future* — donc plus d'information qu'aucun planificateur
n'en aurait — n'extrait rien de plus que « répéter la dernière valeur ».

C'est le même constat que celui du postérieur bayésien, obtenu par une voie
qui ne partage avec lui ni hypothèse de forme, ni méthode d'inférence, ni
code. C'est ce qui le rend intéressant.

### Trois choses que la mesure a appris en chemin

**Un modèle zero-shot n'a pas de R² d'ajustement.** La colonne est vide pour
les trois lignes de fondation, et c'est un résultat : ces modèles n'ajustent
rien, leurs poids sont figés. L'écart ajustement / hors échantillon n'existe
pas, donc il n'y a **aucun surapprentissage à mesurer** — exactement
l'inverse de Banister, dont le meilleur ajustement accompagnait le pire score
hors échantillon.

**L'exogène annoncé par TTM est inerte en zero-shot.** La fiche de
[granite-timeseries-ttm](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3)
annonce l'« intégration de variables exogènes ». Mesure faite : en remplaçant
entièrement les canaux de covariables, la prévision du canal cible est
identique **au bit près** — le dorsal est indépendant par canal et
`enable_forecast_channel_mixing` vaut `False`. Son exogène demande un
fine-tuning. TTM n'est donc enregistré qu'en univarié, et `foundation.py`
refuse `with_covariates=True` pour lui plutôt que de publier deux lignes
qu'on sait égales. Un test le vérifie.

**TTM n'a couvert que 17 plis sur 18.** Sa tête est entraînée pour un horizon
de 96 pas ; le premier pli en demande 152, à cause d'une interruption de
mesure de 135 jours chez cet athlète. Sa ligne ne porte donc pas sur les
mêmes points que les autres. C'est pour cela que la colonne `plis` existe :
sans elle la comparaison paraîtrait à armes égales, et `cli evaluate`
avertit sur la sortie d'erreur quand les plis diffèrent.

### Les deux protocoles ne se comparent pas entre eux

La persistance passe de **+0,318** (protocole en blocs, plus haut) à
**+0,654** ici. Rien ne s'est amélioré : la cible a changé de grille.

L'enveloppe telle que le dépôt la rapportait n'existe qu'aux jours où une
mesure existe — 85 points espacés de 1 à 140 jours. Un bloc de test couvre
alors une durée très variable, et le score agrégé mélange des prévisions à
une semaine et à cinq mois ; comme mesure d'un horizon de prévision, cela ne
veut rien dire. Les modèles de fondation, eux, supposent presque tous un pas
d'échantillonnage constant.

D'où la grille journalière. Ce n'est **pas** une interpolation : un maximum
glissant sur 42 jours est défini pour chaque jour du calendrier à partir des
seules mesures réelles de sa fenêtre, et le code ne le demandait simplement
jamais. Un test l'impose — la sortie journalière, restreinte aux dates
d'observation, est identique à l'ancienne. Les jours où l'enveloppe n'est pas
définie portent `NaN`, jamais la dernière valeur connue : reporter aurait
fabriqué de la donnée et, pire, rendu la cible artificiellement facile à
prédire par persistance.

Cette grille est plus dense et plus autocorrélée (0,974 au retard 1, contre
0,79 pour les 85 points), donc tous les scores montent. Le classement
relatif, lui, est inchangé : aucun modèle de charge ne bat l'inertie.

### Les modèles retenus, et ceux qui ne l'ont pas été

| modèle | taille | licence des poids | covariables zero-shot | exécuté |
|---|---:|---|---|---|
| [Chronos-2](https://huggingface.co/amazon/chronos-2) | 120 M | Apache-2.0 | passées **et futures connues** | **oui** |
| [TTM](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3) (IBM granite) | 855 k | Apache-2.0 | aucune — mesuré | **oui, univarié** |
| [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series) | 11 M | Apache-2.0, **poids sous compte** | oui | non |
| [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | 200 M | Apache-2.0 | aucune | non |
| TimesFM 3.0 | 330 M | **non commerciale** | oui | non |
| [Moirai-2](https://huggingface.co/Salesforce/moirai-2.0-R-small) | 11 M | **CC BY-NC 4.0** | — | non |

Quatre exclusions, quatre raisons distinctes, aucune liée à la performance.

**Moirai-2 et TimesFM 3.0 interdisent l'usage commercial de leurs poids.** Ce
sont deux des modèles que les comparatifs placent en tête. Ce dépôt est sous
MIT ; les citer sans les exécuter est le choix cohérent.

**Les poids de TabPFN ne se téléchargent pas sans compte.** Le code est sous
Apache-2.0, mais `tabpfn` 9.x exige une acceptation de licence interactive et
une clé `TABPFN_TOKEN` obtenue chez Prior Labs. L'adaptateur est écrit et
fonctionnel ; aucun chiffre n'est publié tant qu'il n'a pas tourné.

**TimesFM 2.5 n'accepte aucune covariable**, ce qui le rend inutile pour la
seule question posée ici.

Un détail de version qui compte : le point de contrôle réellement chargé pour
TTM est **R2, pas R3**. `granite-tsfm` 0.3.10, dernière version publiée sur
PyPI, ne sait pas lire les points de contrôle R3 — la moitié des poids, dont
la tête multi-quantile, est réinitialisée au hasard et la sortie n'est pas
finie. La clé de la variante s'appelle `ttm` et non `ttm-r3` pour cette
raison, et un test l'affirme.

### Sur cette machine

Latences mesurées sur un MacBook Pro M3 Pro, 18 Go, **CPU uniquement**,
chargement des poids exclu du chronomètre :

| modèle | par prévision | poids |
|---|---:|---:|
| Chronos-2, avec covariables | 138 ms | 120 M |
| Chronos-2, univarié | 155 ms | 120 M |
| TTM | **3 ms** | 855 k |

Les 3 ms de TTM reproduisent exactement le chiffre publié par IBM. Aucun GPU
n'est nécessaire : le premier chargement de Chronos-2 prend une centaine de
secondes, puis le pipeline est mis en cache et les 18 plis passent en
quelques secondes.

```bash
pip install -e ".[tsfm]"          # torch, chronos-forecasting, granite-tsfm
python -m src.cli evaluate --models chronos2,ttm
```

La commande écrit `data/processed/evaluation_results.parquet` et un JSON de
provenance qui porte les versions de paquets, les révisions de poids, les
paramètres de protocole et les latences. Elle rend les **deux** protocoles :
les six lignes du tableau en tete de ce document, qu'il fallait auparavant
recopier depuis le tableau de bord, en sortent désormais au chiffre près.
C'est aussi ce qui les protège — toute dérive sur ces six lignes signale une
régression de la cible ou du découpage, pas un résultat.

---

## Apparier la charge à la cible, et mesurer ce que le protocole sait voir

Le déséquilibre entre disciplines figurait déjà dans les limites connues de ce
dépôt. Ce qui manquait, c'est l'expérience : sur la fenêtre d'évaluation,

| discipline | part du TRIMP |
|---|---:|
| Bike + VirtualRide | **68,5 %** |
| Run | 30,2 % |
| autres (natation, marche, kayak) | 1,3 % |

alors que l'index de performance est un **rapport de vitesses en course à
pied**. Un modèle à coefficient unique doit donc trouver un `k₁` qui vaille à
la fois pour un kilojoule de vélo et pour un kilomètre de course. Le transfert
vélo → course est partiel et de constante de temps différente, pendant que la
fatigue se cumule pleinement : le coefficient moyenne deux transferts
incomparables et sort zéro.

`build_daily_load` ventile désormais la charge en `trimp_run`, `trimp_bike` et
`trimp_other`, qui somment exactement à `trimp` — la colonne de référence reste
intacte et les six lignes publiées plus haut se reproduisent au chiffre près.

### Sur la cible course : la bonne direction, sans la démonstration

| variante | R² hors éch. (horizon 28 j) | R² ajustement |
|---|---:|---:|
| persistance | **+0,654** | −0,008 |
| **banister course seule** t60/15 | **+0,173** | +0,216 |
| banister course seule t42/12 | +0,124 | +0,139 |
| tendance temporelle | +0,122 | +0,281 |
| prédire la moyenne | +0,090 | 0,000 |
| banister t42/12 (charge totale) | −0,017 | +0,100 |
| banister par sport t42/12 | −0,155 | +0,305 |
| banister + tendance t42/12 | −0,268 | +0,398 |

Restreindre la charge à la course fait passer le modèle de −0,017 à +0,124,
devant « prédire la moyenne ». Mais l'intervalle refuse de conclure :

| variante | contre | gain | IC 95 % | P(gain) |
|---|---|---:|---|---:|
| course seule | charge totale | +0,139 | [−0,117 · +0,395] | 0,852 |
| course seule | prédire la moyenne | +0,038 | [−0,144 · +0,189] | 0,684 |
| course seule | **persistance** | **−1,533** | [−6,279 · +0,048] | 0,032 |

Deux variantes échouent franchement, et c'est instructif. **Un couple de
coefficients par discipline** obtient le meilleur R² d'ajustement du tableau
(+0,305) et l'un des pires hors échantillon (−0,155) : sept paramètres sur
85 points autocorrélés, du surapprentissage fabriqué à la main. **Une tendance
temporelle** mise en concurrence n'absorbe pas la charge, elle ajoute de la
variance (−0,268) — l'athlète passe du vélo à la course en 2018 et toute la
fenêtre est une transition de sport, mais ce confondant n'est pas capté par un
terme linéaire. Il reste ouvert.

### La persistance s'effondre quand l'effort s'allonge

`perf_index` prenait un maximum **sur les durées** avant que l'enveloppe en
reprenne un sur 42 jours : un double cliquet, d'où l'autocorrélation à 0,97 qui
rend la persistance imbattable par construction. Un plafond à 5 minutes dépend
surtout de la puissance anaérobie, un plafond à 45 minutes de l'endurance
aérobie. Les observations par durée existaient déjà dans `mean_max_speed` ;
elles sont désormais publiées une par une.

| durée | obs. | jours | persistance | moyenne | course seule | charge totale |
|---|---:|---:|---:|---:|---:|---:|
| 300 s | 90 | 545 | +0,637 | +0,095 | +0,068 | +0,007 |
| 600 s | 90 | 545 | **+0,697** | +0,122 | +0,204 | +0,192 |
| 1200 s | 89 | 545 | +0,506 | +0,077 | +0,104 | −0,046 |
| 1800 s | 82 | 520 | **+0,170** | +0,028 | +0,081 | −0,279 |
| 2700 s | 59 | 377 | +0,384 | +0,053 | +0,115 | −0,196 |
| 3600 s | 9 | — | écartée : neuf observations ne font pas un R² |

La persistance tombe de +0,697 à 600 s à +0,170 à 1800 s, et c'est le seul
endroit où l'écart avec le modèle de charge cesse d'exclure zéro — à 300 s
l'intervalle vaut [−6,73 · **−0,35**] et P(gain) 0,000, à 1800 s il vaut
[−2,05 · +0,43] et P(gain) 0,398. Les deux modèles deviennent indistinguables
non parce que la charge progresse, mais parce que la persistance s'effondre.

### La cible vélo : l'appariement que le dépôt n'avait jamais fait

La puissance critique du vélo est mesurée depuis 2016 — **417 observations de
`cp20_w`, espacées de deux jours médians, sur trois ans et demi** — et n'avait
jamais servi de cible, parce que `build_perf_observations` filtre sur `is_run`.
C'est pourtant le seul appariement où la discipline qui porte la charge est
celle dont on mesure la performance, avec 4,6 fois plus d'observations que la
course et deux fois plus de plis.

| variante | R² hors éch. (38 plis) | R² ajustement |
|---|---:|---:|
| persistance | **+0,523** | −2,457 |
| moyenne mobile exponentielle | +0,436 | −2,174 |
| **banister vélo seul** t42/12 | **+0,279** | +0,433 |
| banister t42/12 (charge totale) | +0,218 | +0,388 |
| fitness seul t42 | +0,212 | +0,316 |
| prédire la moyenne | −0,090 | 0,000 |
| banister **course** seule t42/12 | **−0,956** | +0,050 |

Et cette fois les intervalles tranchent :

| variante | contre | gain | IC 95 % | P(gain) |
|---|---|---:|---|---:|
| vélo seul | **charge de course** | **+0,631** | **[+0,407 · +0,742]** | **1,000** |
| vélo seul | **prédire la moyenne** | **+0,339** | **[+0,055 · +0,524]** | **0,986** |
| vélo seul | charge totale | +0,078 | [−0,143 · +0,248] | 0,761 |
| vélo seul | persistance | −0,513 | [−2,161 · +0,137] | 0,080 |

Trois choses s'y lisent, et ce sont les premiers résultats de charge de ce
dépôt qui survivent à leur propre intervalle.

**Apparier la discipline de la charge à celle de la cible vaut +0,631 de
compétence, avec P(gain) = 1,000.** C'est aussi un contrôle de validité
discriminante : la charge de course ne prédit *pas* une puissance en vélo
(−0,956), ce qui est la bonne réponse. Si elle l'avait prédite, il aurait fallu
chercher une fuite dans l'alignement des dates avant de crier au transfert
physiologique.

**Le modèle de charge bat « prédire la moyenne » de +0,339, intervalle
[+0,055 · +0,524].** Sur la cible course il ne l'avait jamais fait de façon
démontrable. Il existe donc bien un effet de charge mesurable chez cet athlète
— là où la charge et la performance parlent de la même discipline.

**Il reste battu par la persistance**, mais plus de façon décisive : P(gain)
passe de 0,032 sur la cible course à 0,080 ici, et l'intervalle contient zéro.

### Le plancher de détectabilité, et ce qu'il fait à la conclusion

Tous ces intervalles sont larges, ce qui pose la question que le dépôt n'avait
jamais posée : **quel effet ce protocole saurait-il voir ?** Sans ce nombre,
« on n'a rien trouvé » et « il n'y a rien » sont indistinguables, et le second
ne se déduit pas du premier.

`src/evaluation/sensitivity.py` ajoute à l'enveloppe **réelle** un effet de
charge **synthétique** d'amplitude connue — `fitness − fatigue` sur la charge de
la discipline testée, soit exactement la forme que le modèle postule — puis
repasse le protocole inchangé. `δ` s'exprime en écarts-types d'enveloppe.

| | contre « prédire la moyenne » | contre la persistance |
|---|---|---|
| cible **course** (18 plis) | plancher **0,50 σ** | plancher **2,00 σ** |
| cible **vélo** (38 plis) | **effet déjà vu sans injection** | plancher **1,00 σ** |

Quatre lectures, dont la dernière corrige la façon dont ce dépôt présentait son
propre résultat.

Sur la cible course, le protocole exclut tout effet banisterien supérieur à
**0,5 σ** par rapport à la moyenne — une borne falsifiable, là où « rien
trouvé » n'en est pas une.

Pour battre la persistance sur cette même cible, il faudrait un effet **deux
fois plus dispersé que l'enveloppe entière**. Autrement dit : « aucun modèle de
charge ne bat la persistance » est vrai, mais cette comparaison n'a presque
aucune puissance. La persistance est quasi imbattable sur une cible à 0,97
d'autocorrélation, et l'effet injecté est lui-même lisse — deux moyennes
mobiles exponentielles —, donc la persistance le suit. Le constat tient ; ce
qu'il démontre est bien plus mince qu'il n'y paraissait.

Sur la cible vélo le plancher tombe de moitié, à **1,00 σ**, grâce aux 38 plis
et aux 417 observations. Et contre la moyenne il n'y a plus de plancher à
mesurer : l'effet est détecté **sans aucune injection**, dans les données
réelles.

Le plancher est de surcroît **optimiste** : le signal injecté a précisément la
forme que le prédicteur ajuste, le cas le plus favorable imaginable. Un effet
réel de forme différente devrait être plus grand encore pour être vu.

### Ce que cela change

Le résultat nul de ce dépôt n'est pas annulé, il est **circonscrit**. Il portait
sur une cible de course opposée à une charge aux deux tiers cycliste, mesurée
par un protocole dont on sait maintenant qu'il ne pouvait pas trancher. Apparié
correctement, le modèle de charge montre un effet qui survit à son intervalle.

Reste à savoir si cet effet est un fait ou une particularité de cet athlète :
la section suivante le confronte à treize autres, et la réponse est moins
flatteuse que ce tableau.

```bash
python -m src.cli evaluate       # les cinq protocoles et les quatre planchers
```

---

## Ce que treize autres athlètes en disent

Le résultat ci-dessus tient sur **un athlète et 38 plis**. C'est une anecdote
tant qu'il n'a pas été confronté à d'autres, et le cache OpenData en contient
62. Quinze passent les critères — assez d'observations de puissance, assez
d'historique pour un protocole à horizon — et treize produisent assez de plis
pour compter.

Tout ce dont l'expérience a besoin se trouve dans le `<uuid>.json` de chaque
archive : date, discipline, puissance critique de vingt minutes, et un TRIMP.
Décoder les CSV de 62 athlètes coûterait des heures pour la même réponse. La
contrepartie est déclarée et vérifiée : la charge employée est celle de
GoldenCheetah (`trimp_points`), corrélée à **0,982** avec le TRIMP du dépôt
chez l'athlète de référence (0,984 sur le vélo), pour un rapport médian de
0,938. Suffisant pour savoir si une relation se réplique ; insuffisant pour
publier un coefficient.

### L'effet ne se réplique pas comme fait de population

| comparaison | athlètes | gain médian | Q1 · Q3 | gain > 0 | IC > 0 | IC < 0 | p signe |
|---|---:|---:|---|---:|---:|---:|---:|
| charge vélo / **moyenne** | 13 | −0,008 | [−0,058 · +0,314] | 6/13 | **4** | **0** | 1,000 |
| charge vélo / **persistance** | 13 | −1,376 | [−1,850 · −0,592] | 1/13 | 0 | 6 | **0,003** |
| charge vélo / charge course | 6 | +0,418 | [−0,115 · +0,596] | 4/6 | 2 | 0 | 0,688 |

La médiane est nulle et le test du signe vaut 1,000 : **l'effet mesuré chez
l'athlète de référence n'est pas un fait général.** Il fallait le savoir avant
d'en dépendre.

Mais ce n'est pas du bruit non plus. **Quatre athlètes sur treize ont un
intervalle qui exclut zéro en faveur de la charge, et aucun n'en a un qui
l'exclut contre.** Sous l'hypothèse du hasard, avec des intervalles à 95 %, on
attendrait environ 0,3 de chaque côté. L'asymétrie est le résultat.

| athlète | jours | puiss. | plis | R² charge | R² moyenne | gain | IC 95 % |
|---|---:|---:|---:|---:|---:|---:|---|
| 250ee69b | 437 | 227 | 12 | +0,420 | −0,040 | **+0,442** | **[+0,173 · +0,665]** |
| 9e4573b1 | 1213 | 417 | 38 | +0,283 | −0,090 | **+0,342** | **[+0,060 · +0,526]** |
| 36cdfd30 | 877 | 485 | 27 | +0,408 | +0,128 | **+0,321** | **[+0,181 · +0,458]** |
| bc363cef | 692 | 795 | 19 | +0,172 | +0,009 | **+0,165** | **[+0,020 · +0,290]** |
| 7c3be9dc | 470 | 169 | 11 | +0,272 | −0,061 | +0,314 | [−0,032 · +0,534] |
| 537ba4dd | 688 | 418 | 19 | +0,099 | +0,089 | +0,011 | [−0,279 · +0,158] |
| 1feeb5c6 | 1168 | 601 | 36 | −0,067 | −0,059 | −0,008 | [−0,630 · +0,386] |
| e4dbd3b3 | 672 | 193 | 21 | −0,121 | −0,082 | −0,036 | [−0,448 · +0,176] |
| 9a708c09 | 1116 | 294 | 37 | +0,030 | +0,080 | −0,054 | [−0,598 · +0,328] |
| ea20681e | 968 | 289 | 30 | +0,052 | +0,104 | −0,058 | [−0,223 · +0,176] |
| 62752551 | 752 | 493 | 22 | −0,390 | −0,116 | −0,246 | [−0,991 · +0,320] |
| bdc5457c | 406 | 252 | 14 | −1,718 | −0,925 | −0,412 | [−1,718 · +0,378] |
| 3167e01c | 531 | 264 | 13 | −2,521 | −0,124 | −2,134 | [−18,006 · +0,001] |

`9e4573b1` est l'athlète de référence du dépôt. Il est l'un des quatre
répondeurs — son résultat n'était donc pas un artefact, mais il n'était pas
typique, et tout ce que le dépôt publiait reposait sur un cas favorable choisi
pour d'autres raisons.

### Hétérogénéité réelle, ou défaut de mesure ?

Quand une minorité montre un effet et le reste rien, deux lectures s'opposent
et elles ne mènent pas au même modèle. Soit l'effet est **universel mais noyé**
chez les athlètes mal mesurés — auquel cas leurs estimations ponctuelles
restent positives, seulement avec des intervalles trop larges pour conclure.
Soit l'effet **n'existe que chez certains**, et les estimations des autres se
répartissent autour de zéro quelle que soit leur précision.

| groupe | athlètes | gain médian | gain > 0 | largeur d'IC | puiss. méd. | plis méd. |
|---|---:|---:|---:|---:|---:|---:|
| détectés | 4 | **+0,331** | 4/4 | 0,372 | 451 | 23 |
| non détectés | 9 | **−0,054** | 2/9 | 0,925 | 289 | 21 |

Les détectés sont bien mieux mesurés — intervalles 2,5 fois plus serrés — mais
**leur gain diffère aussi**, et c'est ce qui tranche : les non détectés sont
centrés sur −0,054, avec 2 estimations positives sur 9. Ils ne sont pas
positifs-mais-noyés, ils sont nuls. Le nombre de plis est d'ailleurs presque
identique entre les deux groupes (23 contre 21), donc ce n'est pas une
différence de puissance statistique.

**L'hétérogénéité entre athlètes est réelle.** Chez certains, la puissance de
vingt minutes répond à la charge d'une façon que le modèle de Banister capte ;
chez la plupart, non. Aucune des variables essayées — charge moyenne,
variabilité de la charge, proportion de jours de repos, dispersion de la
puissance — ne sépare les deux groupes sur treize athlètes, et rien ne sera
affirmé là-dessus sans plus de sujets.

### Ce que la réplication change

La domination de la persistance, elle, se réplique solidement : médiane
−1,376, un athlète sur treize au-dessus, test du signe à **p = 0,003**. C'est
le constat le plus robuste de ce dépôt, et il vaut pour la population, pas
seulement pour un cas.

Le reste impose de la prudence. Un jumeau numérique qui suppose une
dose-réponse de charge se trompera chez environ deux athlètes sur trois — et
rien, dans les données dont il dispose à l'inscription, ne permet de savoir
lequel il a en face. Un modèle hiérarchique à pooling partiel est la suite
naturelle : il estimerait la variance inter-athlètes de k₁ au lieu de la
constater, et rendrait les constantes de temps identifiables, ce qu'elles ne
sont pas sur un athlète seul.

```bash
python -m src.cli cohort         # criblage, replication, contraste
```

---

## Le planificateur, rendu correct sous l'incertitude mesurée

Jusqu'ici ce dépôt mesurait. Le planificateur, lui, n'était **importé par
rien** — ni test, ni tableau de bord, ni commande — et son objectif disait :

```python
# optimizer.py, avant
obj = sum(x[j][t] * spec_vals[t] for j in range(7) for t in range(n_types))
model.Maximize(obj)
```

Maximiser la spécificité sous un plafond de volume, c'est prescrire
« entraîne-toi autant que la soupape l'autorise ». C'est défendable si l'on
connaît la dose-réponse de l'athlète. Les treize athlètes du cache disent
qu'on ne la connaît pas : elle existe chez quatre, pas chez neuf, et aucune
variable disponible à l'inscription ne les sépare.

Sous une dose-réponse inconnue, maximiser un proxy est le mauvais côté de
l'incertitude. Le coût d'une charge inutile est réel — blessure, surmenage,
temps — alors que son bénéfice est, chez deux athlètes sur trois, non
mesurable.

### L'objectif retourné

`optimize_week` cherche désormais la charge **minimale** qui délivre le
stimulus demandé. Pour les mêmes contraintes, le même stimulus, et une semaine
écoulée à 250 TRIMP :

| objectif | TRIMP | séances | spécificité |
|---|---:|---:|---:|
| `max_specificity` (historique) | 271 | 6 | 0,43 |
| **`min_load`** (défaut) | **200** | **3** | 0,33 |

26 % de charge en moins et deux fois moins de séances, pour les deux séances
clés exigées dans les deux cas. L'ancien objectif reste accessible par
`--objective max_specificity`, pour que les deux soient comparables plutôt
qu'un remplacé en silence.

L'écart dépend du régime, et il faut le dire : chez l'athlète du dépôt, dont la
dernière semaine connue pèse 446 TRIMP, le plancher de rétention monte à 357 et
les deux objectifs convergent (360 contre 365). La différence se joue quand il
reste de la marge entre le plancher et le plafond — c'est-à-dire précisément
quand un choix est à faire.

Un plancher empêche l'inverse du problème : minimiser sans borne basse
donnerait deux séances clés et cinq jours de repos, ce qui est un minimum au
sens littéral et une régression au sens physiologique. Il vaut
`min_weekly_retention_pct` de la semaine écoulée — dans `params.yaml`, pas
dans le code.

### Trois règles de sécurité qui existaient sans être branchées

**La borne aigu/chronique.** `acute_chronic_ratio` vivait dans
`load_model.py`, sa docstring la présentait comme « la seule règle de sécurité
réellement déclenchable, contrairement au rMSSD et au DFA, elle ne demande que
la charge, qui existe toujours » — et elle n'était appelée nulle part.
`acwr_upper_bound: 1.5` était configuré et jamais lu. La charge de la semaine à
venir, ramenée au jour, est maintenant bornée à 1,5 fois la charge chronique.

Quand ce plafond descend sous le plancher de rétention — un athlète dont la
base chronique s'est effondrée — le plancher est **rabaissé au plafond**, et
non abandonné. L'abandonner ferait rendre un plan à zéro TRIMP : le minimum
littéral, et un conseil inutile. Sur une base de 10 TRIMP/jour, le plan passe
de 400 à 105 TRIMP et le dit.

**Le contrôle de monotonie.** La monotonie de Foster est un rapport
moyenne/écart-type, non linéaire dans les variables de décision, donc
impossible à poser directement dans un solveur linéaire. Son contenu pratique
l'est : une journée légère dans chaque fenêtre glissante, et des bornes de
répétition par type de séance. Sans ces dernières l'optimiseur empilait le
type le moins coûteux qui satisfaisait ses contraintes — il prescrivait
**quatre séances `CALIBRATION`**, qui est un test et non un entraînement. Les
bornes vivent dans `params.yaml`.

**Les indisponibilités déclarées.** `avoid_days` — « je ne peux pas
m'entraîner ce jour-là », extrait du contexte en langage naturel — était posé
en `AddHint`, c'est-à-dire en suggestion que le solveur peut ignorer. Une
indisponibilité déclarée se retrouvait donc planifiée. C'est une contrainte,
et elle passe avant le stimulus prescrit : c'est l'athlète qui connaît son
emploi du temps. Au passage, un `AddHint` que les contraintes rendent
infaisable fait écrire à CP-SAT des pages d'avertissements sur la sortie
standard, ce qui est le symptôme visible du même défaut.

**Le seuil de VFC.** `safety_valve` lisait `planner.rmssd_vma_threshold`, une
clé qui n'existe pas dans `params.yaml`. La valeur configurée sous
`planner.hrv.rmssd_threshold_ms` n'était donc jamais lue, et un défaut codé en
dur gagnait toujours — alors que le commentaire du fichier la présentait comme
« seule source de vérité pour ce seuil ». Les deux valeurs étaient égales par
hasard, ce qui rendait le défaut invisible.

### Le plan dit ce qu'il n'optimise pas

`dose_response.py` mesure la relation charge → performance **chez l'athlète
qu'on a en face**, par le protocole déjà publié — même enveloppe, même
découpage, même bootstrap apparié. Trois verdicts, et non deux :

| verdict | ce qu'il signifie | droit de promettre |
|---|---|---|
| `detectee` | intervalle apparié excluant zéro | parler d'un effet, oui |
| `non detectee` | assez de données, rien à voir | non |
| `indeterminable` | pas assez de données pour regarder | non |

Confondre les deux derniers présenterait une absence de mesure comme une
absence d'effet, qui est exactement l'erreur que ce dépôt s'emploie à ne pas
commettre. Et aucun des trois n'autorise à maximiser une performance prédite :
le verdict n'entre pas dans l'optimisation, il entre dans le rapport.

```
$ python -m src.cli plan

=== Explication du Plan (XAI) ===

Ce plan cherche la charge la PLUS FAIBLE qui délivre le stimulus demandé,
sous les contraintes de sécurité.
Dose-reponse detectee chez cet athlete : competence +0.339 contre « moyenne »,
IC 95 % [+0.055 · +0.524], 38 plis.
Un effet de charge est donc démontré chez cet athlète — ce plan ne le maximise pas
pour autant, il délivre un stimulus au moindre coût.
```

Le verdict reproduit au chiffre près celui de la cohorte, parce qu'il passe
par le même code. Si les deux divergeaient, l'un des deux mentirait.

### Un plan dégradé ne se lit plus comme un plan optimal

L'ancien code retombait silencieusement sur l'heuristique dès qu'une
contrainte rendait le problème infaisable. `WeeklyPlan` porte maintenant des
`notes`, et le rapport les publie : plancher rabaissé, stimulus abandonné,
borne aigu/chronique inactive faute d'historique. Une borne silencieusement
inactive est pire qu'une borne absente.

```bash
python -m src.cli plan                              # charge minimale suffisante
python -m src.cli plan --objective max_specificity  # comportement historique
```
