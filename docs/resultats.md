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

## Ce que soixante-et-un athlètes en disent

Le résultat de l'athlète de référence tenait sur 38 plis chez une personne.
`cli fetch-athletes` a porté le cache de 62 à **183 archives** ; 61 athlètes
passent les critères de la cohorte et portent l'expérience.

Tout vient des `<uuid>.json` des archives : date, discipline, puissance
critique de vingt minutes, TRIMP. La contrepartie est déclarée et vérifiée — la
charge employée est `trimp_points` de GoldenCheetah, corrélée à **0,982** avec
le TRIMP du dépôt chez l'athlète de référence (0,984 sur le vélo), pour un
rapport médian de 0,938. Suffisant pour savoir si une relation se réplique ;
insuffisant pour publier un coefficient.

### Trois résultats, et le nombre de sujets change les conclusions

| comparaison | athlètes | gain médian | Q1 · Q3 | gain > 0 | IC > 0 | IC < 0 | p signe |
|---|---:|---:|---|---:|---:|---:|---:|
| charge vélo / **moyenne** | 61 | +0,130 | [−0,058 · +0,366] | 40/61 | **23** | **5** | **0,020** |
| charge vélo / **persistance** | 61 | −1,153 | [−1,707 · −0,592] | 3/61 | 0 | **32** | **<0,0001** |
| charge vélo / **charge course** | 35 | +0,481 | [+0,160 · +0,731] | 29/35 | **14** | 0 | **0,0001** |

**L'appariement de discipline est le résultat le plus solide du dépôt.** Sur
les 35 athlètes qui pratiquent les deux, opposer à une puissance vélo la charge
vélo plutôt que la charge course vaut +0,481 de compétence médiane, avec
quatorze intervalles excluant zéro et **aucun en sens inverse**, test du signe
à p = 0,0001. La charge de course sert ici de contrôle négatif : si elle
prédisait une puissance en vélo, il faudrait chercher une fuite dans
l'alignement des dates avant de parler de transfert physiologique.

**La domination de la persistance est écrasante.** Trois athlètes sur 61 la
battent, trente-deux ont un intervalle qui l'exclut en sa faveur, p < 0,0001.
C'est le constat le plus robuste du projet, et il vaut pour la population.

### Ce que treize athlètes ne pouvaient pas voir

Sur les treize premiers athlètes, le test du signe contre la moyenne valait
**p = 1,000** et j'en concluais que l'effet ne se répliquait pas comme fait de
population. Sur 61, il vaut **p = 0,020**.

Cette conclusion n'était pas fausse, elle était **sans puissance** — exactement
ce que le plancher de détectabilité construit plus haut sert à chiffrer. Le
nombre de sujets était le facteur limitant, et il l'était de façon mesurable.

Deux choses apparaissent aussi, que treize sujets cachaient. D'abord **cinq
athlètes dont l'intervalle exclut zéro *contre* la charge** : chez eux, nourrir
le modèle de la charge dégrade la prévision par rapport à prédire la moyenne.

| athlète | jours | puiss. | plis | R² charge | R² moyenne | gain | IC 95 % |
|---|---:|---:|---:|---:|---:|---:|---|
| a49a2a94 | 500 | 275 | 12 | −0,777 | −0,044 | −0,703 | [−1,404 · −0,210] |
| 0031326c | 456 | 263 | 11 | −0,359 | +0,048 | −0,428 | [−1,150 · −0,018] |
| 2d6239bd | 617 | 178 | 18 | −0,486 | −0,047 | −0,419 | [−0,849 · −0,048] |
| 7b8034de | 648 | 1090 | 17 | −0,362 | −0,089 | −0,250 | [−0,500 · −0,042] |
| ae4b2ab7 | 1245 | 276 | 47 | −0,126 | −0,056 | −0,066 | [−0,216 · −0,012] |

L'hétérogénéité est donc **bidirectionnelle**, et non une simple présence ou
absence d'effet. Un jumeau numérique qui suppose une dose-réponse ne se
contente pas de ne rien apporter chez une majorité : chez une minorité, il
prédit activement moins bien que la moyenne.

### Hétérogénéité réelle, ou défaut de mesure ?

| groupe | athlètes | gain médian | gain > 0 | largeur d'IC | puiss. méd. | plis méd. |
|---|---:|---:|---:|---:|---:|---:|
| détectés | 23 | **+0,442** | 23/23 | 0,397 | 408 | 25 |
| non détectés | 38 | **−0,025** | 17/38 | 0,688 | 294 | 22 |

Le contraste tient à 61 sujets comme il tenait à 13, et il est plus net. Les
détectés sont mieux mesurés — intervalles 1,7 fois plus serrés — mais **leur
gain diffère aussi**, et c'est ce qui tranche : les non détectés sont centrés
sur −0,025, avec 17 estimations positives sur 38, soit la répartition qu'on
attend autour de zéro. Ils ne sont pas positifs-mais-noyés, ils sont nuls.
Le nombre de plis médian est d'ailleurs presque identique entre les deux
groupes (25 contre 22), donc ce n'est pas une différence de puissance.

**L'hétérogénéité entre athlètes est réelle.** Et aucune variable disponible à
l'inscription ne sépare les deux groupes, ce qui reste la question ouverte du
projet.

---

## Les cinq athlètes chez qui la charge nuit

C'est le résultat le plus inattendu de la réplication : chez cinq athlètes sur
soixante-et-un, l'intervalle apparié exclut zéro **contre** la charge — le
modèle de Banister y prédit moins bien que « prédire la moyenne ». Un artefact
non instruit à cet endroit fausserait la lecture de tout le reste, donc les
cinq cas ont été ouverts un par un.

| athlète | gain | IC | puissances | jours | plis |
|---|---:|---|---:|---:|---:|
| `a49a2a94` | −0,703 | [−1,404 · −0,210] | 275 | 501 | 12 |
| `0031326c` | −0,428 | [−1,150 · −0,018] | 263 | 490 | 11 |
| `2d6239bd` | −0,419 | [−0,849 · −0,048] | 178 | 725 | 18 |
| `7b8034de` | −0,250 | [−0,500 · −0,042] | 1090 | 653 | 17 |
| `ae4b2ab7` | −0,066 | [−0,216 · −0,012] | 276 | 1662 | 47 |

### Aucune signature d'artefact

Trois hypothèses d'artefact ont été testées contre le **taux de base** des 23
athlètes détectés — sans groupe de comparaison, n'importe quelle valeur a l'air
suspecte.

**Trou de données :** écart médian entre observations de 1 à 3 jours, trou
maximal de 17 à 127 jours, contre 1 jour et 56 jours en médiane chez les
détectés. Rien d'anormal.

**Changement de capteur, par rupture de niveau.** On cherche la coupure qui
maximise l'écart de moyenne entre les deux côtés de la série de puissance.
Quatre des cinq restent sous le 9ᵉ décile des détectés (1,28 σ) ; `ae4b2ab7`
sort à 1,54 σ, au-dessus de ce décile mais **sous le maximum des détectés**
(1,68 σ). Suspect, pas concluant.

**Changement d'échelle, par rapport des écarts-types de part et d'autre de la
coupure.** `2d6239bd` sort à 0,55, juste sous le minimum des détectés (0,59) :
sa puissance perd la moitié de sa dispersion en milieu de série, ce qui est la
signature d'un changement d'home-trainer. Les résumés publics ne portent pas le
matériel, donc cela reste une hypothèse non vérifiable à partir de ces données.

### L'hypothèse mécaniste que j'avais, et sa réfutation

Les cinq négatifs se distinguent nettement sur un point : **leur charge monte
beaucoup plus fort** — +18,6 TRIMP/an contre +4,2 chez les détectés. D'où une
hypothèse plausible : si la charge cumulée tend fortement, la convolution tend
aussi, le modèle extrapole la performance vers le haut hors échantillon, et
dépasser la cible est pire que prédire la moyenne.

Elle est fausse. Mesuré sur les 61 athlètes, **le modèle ne surestime pas** :
biais médian −0,045 σ, 27 athlètes surestimés sur 61 — l'erreur est
symétrique. Et le biais ne dit rien du score : ρ(biais, gain) = **−0,018** sur
61 sujets, soit un nul bien alimenté, pas une absence de puissance.

Ce qui gouverne le score est l'**amplitude** de l'erreur, pas son signe :
ρ(erreur absolue, gain) = −0,720. Les cinq négatifs ne se trompent pas de
direction, ils se trompent plus — 1,045 σ d'erreur absolue contre 0,631 chez
les détectés. C'est une bonne nouvelle pour le planificateur : un modèle dont
les erreurs sont symétriques ne promet pas systématiquement trop.

### Les deux écarts qui restent sont juste sous le plancher

Les deux différences observées — pente de charge et erreur absolue — vont dans
le sens de l'explication, et aucune des deux n'est démontrable à cinq sujets.
Test de permutation, 20 000 tirages, médiane des cinq contre celle des 56
autres :

| grandeur | écart observé | p | écart détectable à 5 % |
|---|---:|---:|---:|
| pente de charge (TRIMP/an) | +14,0 | 0,074 | **+16,9** |
| erreur absolue (σ) | +0,249 | 0,070 | **+0,267** |
| biais signé (σ) | −0,086 | 0,761 | +0,577 |

Les deux premiers écarts sont réels et **plus petits que ce que ce protocole
sait voir**. Il manquait environ 20 % d'amplitude, ou des sujets. C'est la
forme exacte du résultat que le plancher de détectabilité existe pour produire
— et c'est la troisième fois dans ce dépôt qu'un constat négatif est en réalité
un constat sous-dimensionné.

### Ce qu'on peut en dire

**Aucun artefact établi, aucun mécanisme établi.** Un cas (`2d6239bd`) porte
une rupture d'échelle compatible avec un changement de matériel, un autre
(`ae4b2ab7`) une rupture de niveau à la limite du taux de base ; les trois
autres ressemblent aux athlètes détectés sur tout ce qui est mesurable.
L'hétérogénéité bidirectionnelle n'est donc pas un défaut de mesure
identifiable, et le planificateur garde raison de ne rien promettre à un
athlète dont la dose-réponse n'est pas détectée — y compris quand elle est
détectée dans le mauvais sens.

---

## Ce que la mise en commun des athlètes ajoute

Rejouer le protocole athlète par athlète compte des verdicts : vingt-trois
sujets sur soixante-et-un montrent un effet, trente-trois non, cinq le voient
se retourner contre eux. Cela ne dit pas **de combien** ils diffèrent — un
test du signe ne mesure pas une dispersion. Un modèle hiérarchique l'estime,
et partage les constantes de temps pour qu'elles soient contraintes par
soixante-deux historiques au lieu d'un.

`python -m src.cli pool` — **62 athlètes, 1816 points d'enveloppe** (un tous
les 28 jours), quatre chaînes de 400 + 400 itérations en **2 h 26**,
**zéro divergence**, `r_hat` ≤ 1,01.

| paramètre | moyenne | écart-type | HDI 94 % | ESS |
|---|---:|---:|---|---:|
| **τ₁ (fitness, jours)** | **70,3** | 10,1 | **[52,0 · 88,4]** | 295 |
| **τ₂ (fatigue, jours)** | **29,7** | 3,7 | **[23,3 · 37,0]** | 733 |
| μ effet net | **+0,610** | 0,047 | **[+0,521 · +0,696]** | 708 |
| **σ effet net** | **0,205** | 0,040 | **[+0,131 · +0,278]** | 609 |
| μ gain fatigue | −0,204 | 0,138 | [−0,453 · +0,068] | 535 |
| σ gain fatigue | 0,586 | 0,120 | [+0,374 · +0,807] | 795 |
| μ tendance | −0,024 | 0,049 | [−0,112 · +0,070] | 764 |
| **σ tendance** | **0,334** | 0,040 | **[+0,261 · +0,409]** | 694 |

Les gains s'expriment en écarts-types d'enveloppe par écart-type de
`fitness − fatigue` : chaque cible est centrée-réduite sur son propre
historique, sans quoi mettre en commun un gain reviendrait à mélanger des
watts et non des effets.

### L'hétérogénéité est maintenant établie — correction d'un résultat publié ici

`σ` de l'effet net vaut **0,205, HDI [+0,131 · +0,278]**. L'intervalle
**exclut zéro**.

À quatorze athlètes, ce même paramètre sortait à 0,212 **[0,000 · 0,393]** et
cette page concluait que l'hétérogénéité n'était pas établie. La moyenne n'a
pas bougé — 0,212 puis 0,205 — **seul l'intervalle s'est resserré**, de 0,39
de large à 0,15. La conclusion précédente n'était donc pas fausse, elle était
**sous-dimensionnée** : il manquait des sujets, pas un modèle différent. C'est
le scénario exact que le plancher de détectabilité existe pour anticiper, et
il vaut d'être écrit en clair plutôt que corrigé en silence.

Les athlètes diffèrent donc réellement, d'environ un tiers de l'effet moyen en
écart-type. Un effet de population de +0,61 avec une dispersion de 0,21 place
la grande majorité des sujets du bon côté de zéro — ce que l'ajustement montre
— mais rend un athlète sur vingt proche de l'indifférence.

### Les constantes de temps sont identifiées, et elles contredisent la littérature

C'était l'objectif principal de la mise en commun. Sur un athlète seul, le
postérieur du dépôt ne resserre τ₁ que de **41 %** par rapport à son prior :
les données ne l'identifient pas. Partagée sur soixante-deux historiques, elle
sort à **70,3 jours [52,0 · 88,4]** et τ₂ à **29,7 [23,3 · 37,0]**.

Ce sont des intervalles informatifs — et c'est là que la bonne nouvelle
s'arrête. La littérature place τ₁ vers 42 jours et τ₂ vers 10–15. Le prior de
τ₂ est une normale tronquée de moyenne 12 et d'écart-type 5 : le postérieur
atterrit **3,5 écarts-types de prior au-dessus**, en poussant contre lui. À
quatorze athlètes l'estimation était compatible avec la littérature (51,7 et
19,0) ; à soixante-deux, les deux constantes ont dérivé vers le haut en se
resserrant. Un paramètre qui lutte contre son prior et s'en éloigne quand on
ajoute des données ne se lit pas comme une mesure physiologique.

L'explication la plus probable est dans la cible, pas dans les athlètes :
**l'enveloppe est un maximum glissant sur 42 jours, lu tous les 28 jours.**
C'est une série très lissée par construction, et une cible lissée préfère des
noyaux de convolution lents — le modèle apprend en partie la fenêtre de
lissage. La vérification à faire est directe : refaire l'ajustement à
fenêtre d'enveloppe réduite et voir si τ₁ et τ₂ suivent la fenêtre. S'ils la
suivent, ces constantes mesurent le protocole et non la physiologie, et la
ligne la plus citable de ce tableau tombe.

Les constantes sont donc **identifiées sans être validées** — deux choses
différentes, et seule la première était l'objectif annoncé.

### La charge n'hérite pas d'une tendance partagée

Un premier ajustement, sans terme de tendance, laissait τ₁ fuir à **127 jours**
— 4,6 écarts-types au-delà de son prior. À cette constante, la charge cumulée
*est* une tendance, et le gain mesurait l'une en croyant mesurer l'autre.

Avec une tendance concurrente par athlète, le verdict se maintient à
soixante-deux sujets : **μ tendance = −0,024 [−0,112 · +0,070]**, centré sur
zéro et désormais serré, alors que **σ tendance = 0,334 [+0,261 · +0,409]**
exclut zéro. Les athlètes ont de vraies tendances, dans les deux sens, qui
s'annulent en moyenne — et le terme de charge ne capte pas une dérive commune.
Le contrôle passe, et il passe mieux qu'à quatorze.

### Et voici la signature du surapprentissage, chiffrée

| | athlètes avec un effet excluant zéro |
|---|---|
| ajustement hiérarchique (**dans** l'échantillon) | **61 / 62** |
| protocole à horizon (**hors** échantillon) | **23 / 61** |

Ce n'est pas une contradiction, c'est l'observation fondatrice du dépôt mise en
chiffres : **ajuster n'est pas prédire.** Le même modèle, sur les mêmes
athlètes, explique la performance passée chez tous sauf un et n'en prévoit la
suite que chez un peu plus d'un tiers. Un postérieur bayésien favorable ne vaut
donc rien comme preuve de capacité prédictive — et c'est exactement pour cela
que ce dépôt ne publie jamais un ajustement sans son pendant hors échantillon.

L'écart s'est creusé en passant de quatorze à soixante-deux athlètes
(12/14 contre 4/13 auparavant). La mise en commun resserre les gains vers la
moyenne de population : plus la cohorte est grande, plus un athlète mal mesuré
est tiré vers +0,61, et plus son intervalle d'ajustement exclut zéro sans que
rien n'ait été prédit chez lui.

### Les deux protocoles s'accordent en rang, pas en signe

Les deux mesures portent sur les mêmes 61 athlètes et peuvent donc être
croisées. La corrélation entre le gain hiérarchique et le `skill` hors
échantillon est de **+0,406** : les deux protocoles classent les athlètes dans
le même ordre.

| groupe hors échantillon | n | gain hiérarchique moyen |
|---|---:|---:|
| la charge aide (IC > 0) | 23 | **+0,669** |
| non détecté | 33 | +0,601 |
| la charge dégrade (IC < 0) | 5 | **+0,429** |

Le seul athlète des soixante-deux dont le gain d'ajustement n'exclut pas zéro
(`ae4b2ab7`, +0,281 [−0,021 · +0,558]) est aussi l'un des cinq chez qui la
charge dégrade la prévision. L'accord est réel.

Mais **l'ajustement hiérarchique ne produit aucun gain négatif**, pas même chez
ces cinq-là. C'est le retrait vers la moyenne qui fait cela, et c'est le prix
déclaré de la mise en commun : elle emprunte de la force aux autres athlètes,
donc elle refuse de croire un sujet isolé qui va à l'inverse de la population.
Si ces cinq cas sont réels, ce modèle est structurellement incapable de les
voir — et c'est le protocole hors échantillon, lui seul, qui les signale.

### Ce que la puissance de mesure explique, et ce qu'elle n'explique pas

La question ouverte du projet — **qu'est-ce qui distingue un répondeur ?** —
reçoit ici son premier élément de réponse, partiel.

| groupe | jours d'historique (médiane) | plis | largeur d'IC (médiane) |
|---|---:|---:|---:|
| la charge aide | 851 | 25 | **0,397** |
| non détecté | 717 | 22 | 0,678 |
| la charge dégrade | 617 | 17 | 0,801 |

Les répondeurs sont les athlètes les mieux mesurés : la corrélation de rang
entre `skill` et longueur d'historique est de **+0,28**, et l'intervalle des
non-détectés est **deux fois plus large** que celui des détectés. Une partie
de la non-détection est donc un manque de puissance, pas une absence d'effet.

Une partie seulement. Le même athlète `ae4b2ab7` a **1245 jours et 47 plis** —
le mieux mesuré du groupe négatif, et l'intervalle le plus serré des cinq
(`[−0,216 · −0,012]`). Sa non-réponse ne s'explique pas par le bruit. Et
l'historique n'est de toute façon pas une variable disponible à l'inscription :
il décrit ce qu'on a observé, pas ce qu'est l'athlète.

**Rien de ce qui est connu au premier jour ne sépare encore les groupes.** La
question reste ouverte — mais elle est passée d'inattaquable à mesurable.

### Deux réserves de protocole

**62 athlètes ici, 61 dans le protocole à horizon.** Le critère d'inclusion
n'est pas le même : ce modèle demande des points d'enveloppe, pas des plis de
validation. Les totaux ne se comparent pas ligne à ligne ; les 61 athlètes
communs, eux, se croisent bien, et c'est ce croisement qui est publié ci-dessus.

**Un point tous les 28 jours.** L'enveloppe est un maximum glissant sur
42 jours, donc très redondante au pas journalier, et le noyau de convolution
coûte `n_points × n_jours` par athlète. C'est la même grandeur, lue moins
souvent — mais c'est un choix de coût, il est déclaré, et la section sur les
constantes de temps explique pourquoi il n'est peut-être pas neutre.

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

---

## Les règles du planificateur, confrontées aux données

Le planificateur imposait trois bornes — ACWR sous 1,5, un plancher de
retention, au plus trois jours d'entraînement de suite — qui venaient de la
littérature et de l'usage. C'est une raison de les écrire, pas une mesure.
Aucune n'avait jamais rencontré de données, alors que le dépôt exige une
validation hors échantillon de tout le reste.

`python -m src.cli validate-plan` — **66 athlètes, 6768 semaines
calendaires.** Pour chaque semaine : sa charge, son ACWR en fin de semaine, sa
plus longue série de jours consécutifs, et son issue — la variation du meilleur
effort entre `[fin−42 ; fin−7]` et `[fin+7 ; fin+42]`, réduite par l'écart-type
de la puissance de cet athlète. La comparaison respect / violation se fait **à
l'intérieur de chaque athlète**, puis s'agrège par un test du signe, comme la
cohorte.

### Le verdict

| borne | régime | statistique | athlètes | écart | en faveur | p | verdict |
|---|---|---|---:|---:|---:|---:|---|
| ACWR ≤ 1,5 | toutes semaines | meilleur | 55 | −0,333 | 7/55 | <0,0001 | contredite |
| ACWR ≤ 1,5 | toutes semaines | médian | 55 | −0,104 | 18/55 | 0,0145 | contredite |
| **ACWR ≤ 1,5** | **référence ≥ 15 obs** | **meilleur** | 14 | −0,014 | 8/14 | 0,791 | **indéterminable** |
| **ACWR ≤ 1,5** | **référence ≥ 15 obs** | **médian** | 14 | −0,042 | 4/14 | 0,180 | **indéterminable** |
| **ACWR ≥ 0,8** | **référence ≥ 15 obs** | **meilleur** | 55 | **+0,085** | 40/55 | **0,0010** | **soutenue** |
| **ACWR ≥ 0,8** | **référence ≥ 15 obs** | **médian** | 55 | **+0,051** | 37/55 | **0,0145** | **soutenue** |
| ≤ 3 jours consécutifs | référence ≥ 15 obs | médian | 37 | −0,013 | 15/37 | 0,324 | indéterminable |

**Le plancher de retention est la seule borne soutenue.** Les semaines dont
l'ACWR reste au-dessus de 0,8 sont suivies d'un meilleur effort : +0,051 σ sur
la statistique insensible au nombre de tirages, 37 athlètes sur 55 en faveur,
p = 0,0145. Elle survit aux deux contrôles et aux deux statistiques, et elle
est détectée **sans injection**. « Ne pas prescrire le désentraînement » est
donc une règle mesurée, pas une convention.

**Le plafond d'ACWR n'est ni soutenu ni réfuté**, avec un plancher de
détectabilité de **0,20 σ** : son effet réel, s'il existe, est plus petit que
ce que 14 athlètes comparables permettent de voir.

**Le plafond de jours consécutifs ne conclut pas** (p = 0,324). La ventilation
est pourtant suggestive : l'issue monte de 0 à 3 jours consécutifs
(−0,025 → +0,091 σ) puis retombe à partir de 4 (−0,005). Le seuil de 3 est
donc posé exactement au sommet — mais la contrainte « ≤ 3 » mélange 0, 1, 2 et
3 jours, ce qui dilue le gain qu'elle devrait capter. C'est un défaut de
formulation de la contrainte, pas du seuil.

### Comment j'ai failli publier la réfutation d'une règle de sécurité

Les deux premières lignes du tableau sont à lire comme un avertissement. Le
protocole naïf dit que **respecter la borne haute d'ACWR coûte 0,333 σ**, avec
7 athlètes sur 55 en faveur et p < 0,0001 — un résultat massif, net, et faux.
Trois confonds s'y empilaient, et chacun a demandé d'être trouvé avant d'être
corrigé.

**L'enveloppe ne peut pas servir d'issue.** Premier essai : variation de
l'enveloppe sur 28 jours. L'enveloppe est un maximum glissant sur 42 jours,
donc elle ne bouge que par paliers, et sa variation médiane valait **exactement
zéro chez les soixante-six athlètes**. Le test du signe tournait sur des
asymétries de bruit.

**La semaine jugée contaminait les deux bornes de la différence.** Un maximum
glissant qui regarde 42 jours en arrière englobe la semaine qu'on évalue :
l'effort dur de cette semaine *est* une observation de performance. D'où un
tampon de 7 jours de part et d'autre, qui exclut la semaine des deux fenêtres.

**Un maximum sur plus de tirages est plus grand.** Les semaines à ACWR élevé
sont suivies de plus de sorties : dans la tranche au-delà de 2,0, la fenêtre
d'issue compte **5,2 observations de plus** que celle de référence. Le maximum
monte sans qu'aucune performance ait progressé. Chaque borne est donc jugée
aussi sur l'effort **médian** de la fenêtre, insensible au nombre de tirages —
et ce seul contrôle divise l'effet par trois, de −0,333 à −0,104.

**Et le dernier confond ne se contrôle pas, il s'exclut.** Un ACWR élevé
signifie souvent que la charge chronique était basse : l'athlète revenait d'un
arrêt. La fenêtre de référence est alors elle-même déprimée, et la
« progression » mesurée est un retour à la normale. Les chiffres le disent sans
ambiguïté — la tranche au-delà de 2,0 ne compte que **8,1 observations dans la
fenêtre de référence contre 13,3 après**. En exigeant 15 observations avant la
semaine jugée, donc un entraînement régulier, l'écart tombe à **−0,042 et le
test ne conclut plus**. Le gradient par tranche s'aplatit sur toute l'étendue
de l'ACWR : −0,01 σ sous 0,8, +0,10 au-delà de 2,0.

Publier la première version aurait signifié démonter une borne de sécurité sur
un artefact d'échantillonnage additionné d'un biais de sélection. Le prix du
contrôle est visible : on passe de 55 athlètes comparables à 14, et le plancher
monte à 0,20 σ. C'est le bon prix.

### Ce que ce protocole ne fait pas

Il ne reconstruit pas les **types de séances** — les résumés publics ne les
portent pas — donc les plafonds par type ne sont pas testables ici. Il
n'établit **aucune causalité** : on compare les semaines d'un même athlète, pas
des bras tirés au sort. Et le plancher de retention du planificateur porte sur
80 % de la charge d'une semaine de référence, quantité qui n'existe pas dans un
historique observé ; `ACWR ≥ 0,8` en est l'analogue le plus proche, et c'est
une approximation déclarée, pas la règle elle-même.

---

## Qui répond ? La question ouverte, posée hors échantillon

L'effet de charge existe chez 23 athlètes sur 61, est indétectable chez 33 et
dégrade la prévision chez 5. C'est le résultat le plus utile du dépôt et le
plus inexploitable : savoir qu'un athlète sur trois répond ne dit pas **lequel**,
et un planificateur a besoin du second.

### Pourquoi la meilleure corrélation n'est pas une réponse

Chercher parmi douze prédicteurs celui dont la corrélation avec le gain est la
plus forte, puis le publier, c'est tester douze hypothèses et n'en déclarer
qu'une. Le dépôt exige du hors échantillon partout ailleurs. La question
devient donc : **en retirant un athlète, les soixante autres permettent-ils de
dire s'il répond ?**

`python -m src.cli responders` — centroïde par classe sur prédicteurs
centrés-réduits, un athlète retiré à la fois, **le centrage et les centroïdes
recalculés dans chaque pli**. Les calculer une fois sur tout l'échantillon
ferait fuir la cible par le prétraitement, ce qui est la façon la plus discrète
de fabriquer une performance hors échantillon.

| famille de prédicteurs | prédicteurs | athlètes | exactitude | classe majoritaire | p | plancher |
|---|---:|---:|---:|---:|---:|---:|
| connu au premier jour | 2 | 55 | 52,7 % | 60,0 % | 0,892 | **0,05 σ** |
| dérivé de l'historique | 11 | 61 | 55,7 % | 62,3 % | 0,882 | non atteint |
| tout | 13 | 55 | 47,3 % | 60,0 % | 0,980 | 0,05 σ |

**Rien ne sépare les répondeurs.** Les trois familles font moins bien que
« prédire systématiquement la classe majoritaire ». La famille complète fait
même nettement moins bien que celle à deux prédicteurs — signature classique de
trop de variables pour trop peu de sujets.

### Les deux nuls ne se valent pas

C'est l'intérêt du plancher. Pour la famille **connue au premier jour**, un
décalage de 0,05 écart-type des prédicteurs des répondeurs suffit à être
détecté : le test est sensible, et son nul est **bien alimenté**. Rien de ce
qu'on sait d'un athlète avant sa première séance ne le distingue — c'est une
vraie réponse à la question ouverte du projet, et elle est négative.

Pour la famille **dérivée de l'historique**, le plancher n'est pas atteint même
à 0,40 écart-type : onze prédicteurs sur 61 sujets diluent le signal dans la
distance euclidienne. « L'historique ne prédit pas la réponse » n'est donc
**pas établi** — ce test ne saurait pas le voir.

### Les corrélations, avec le seuil que l'échantillon sait distinguer

| prédicteur | ρ | p permuté | ρ détectable |
|---|---:|---:|---:|
| `jours` (étendue d'historique) | **+0,236** | 0,064 | 0,250 |
| `n_observations` | +0,179 | 0,166 | 0,252 |
| `age` | +0,132 | 0,333 | 0,265 |
| `jours_actifs_pct` | +0,113 | 0,385 | 0,249 |
| `cv_charge_hebdo_pct` | −0,165 | 0,205 | 0,252 |
| `pente_charge_an` | −0,129 | 0,322 | 0,252 |

Aucun ne franchit son propre seuil, et le seuil de Bonferroni pour douze
prédicteurs testés est à **p < 0,0042**. L'étendue d'historique reste le
meilleur candidat — et ce n'est pas une variable connue à l'inscription, donc
même vraie elle ne résoudrait pas le démarrage à froid.

### Deux défauts que le résultat a révélés

**Un prédicteur constant ressortait « significatif ».** `sexe_feminin` vaut 0
chez les soixante-et-un athlètes du cache : le jeu de données GoldenCheetah
OpenData, tel que ce cache l'a échantillonné, ne contient que des hommes. Une
colonne constante n'a pas de corrélation **définie**, le p permuté d'un NaN
ressortait à 0,0000, et ce prédicteur dégénéré était **le seul déclaré retenu
de la première exécution**. Il est maintenant écarté — et rapporté comme
écarté, parce qu'un tableau qui ne dit pas ce qu'il a retiré se lit comme si
tout avait été testé.

**Six athlètes avaient un âge négatif.** Ils déclarent une année de naissance
postérieure à la publication du jeu de données, d'où des âges de **−58 à
−42 ans**. Leur présence portait la corrélation de l'âge à **+0,230**, la
deuxième du tableau ; une fois écartés, elle tombe à **+0,132**. Une borne de
plausibilité explicite les retire désormais, et le compte des athlètes passe de
61 à 55 sur les familles qui utilisent l'âge — ce que le tableau affiche.

### Ce qui reste ouvert, et ce qui ne l'est plus

**Fermé :** rien de ce qu'on sait d'un athlète le premier jour ne permet de
prévoir s'il répondra à la charge. Le test est sensible et le nul tient.

**Ouvert :** l'historique pourrait prédire sans que ces 61 athlètes puissent le
montrer. La voie n'est pas plus de prédicteurs — elle en a déjà trop pour
l'échantillon — mais plus de sujets, ou un prédicteur postulé à l'avance plutôt
que cherché parmi douze.
