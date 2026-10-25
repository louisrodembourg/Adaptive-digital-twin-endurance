# Méthode

Comment les chiffres sont produits, ce qui est réel, et les règles que le code
applique pour qu'un résultat ne puisse pas être plus beau qu'il n'est.

Présentation courte : [README](../README.md).
Les mesures elles-mêmes : [résultats](resultats.md).

---

## Ce qui est réel, et ce qui ne l'est pas

**Réel.** 917 séances d'un athlète de GoldenCheetah OpenData, d'août 2014 à
janvier 2020 : 648 vélo, 172 course, 49 sans discipline, 23 home-trainer,
21 natation. Aucune donnée personnelle, aucune coordonnée GPS.

**Validé contre une implémentation externe.** GoldenCheetah publie son propre
TRIMP. Régression du nôtre contre le sien sur 863 séances : **pente 0,934,
ordonnée +1,05, r = 0,982, écart-type résiduel 15,6**. La pente compte
davantage que la corrélation — elle dit que notre échelle est 7 % plus basse.
Un coefficient de rang aurait donné 0,97 en masquant complètement cet écart,
alors que le TRIMP est précisément une grandeur d'échelle qui alimente une
convolution linéaire.

**Estimé, pas supposé.** FC max 194 bpm et FC repos 62 bpm, dérivées de
l'historique. La version précédente supposait 205 et 65, annotées « à
calibrer », alors que tout le TRIMP en dépend.

**Absent, et non simulé.** Ni ce jeu de données ni les fichiers `.fit` d'une
montre Coros ne contiennent d'intervalles RR. Le DFA α₁ renvoie donc `NaN`,
et deux des règles de sécurité n'ont aucune source. Une version antérieure
fabriquait ces intervalles par un processus AR(1) à graine fixe et s'en
servait pour bloquer des séances à haute intensité ; le générateur vit
désormais dans `tests/fixtures/`, sous un nom qui dit ce qu'il produit.

**Extrapolé, et signalé comme tel.** Les prédictions de chrono au-delà de
45 minutes extrapolent une loi de Riegel ajustée sur des efforts de 5 à
45 minutes (exposant 0,0771, R² 0,987). Aucun semi-marathon ne figure dans
les données de cet athlète.

---

## Limites connues

**Le vélo pèse 80 % de la charge totale, la course seulement 13 %.** La
course est longtemps restée la seule discipline où la performance servait de
cible : le modèle devait deviner une capacité de course à partir d'une charge
très majoritairement cycliste. La puissance critique du vélo était mesurée
depuis 2016 mais jamais évaluée, `build_perf_observations` filtrant sur
`is_run` ; elle sert désormais de cible à part entière, et l'écart mesuré
entre les deux appariements est devenu le résultat central de ce dépôt. Le
déséquilibre subsiste tel quel **sur la cible course**, lui.

**Le modèle n'est pas identifié.** k₁ et k₂ sont corrélés à 0,986 dans le
postérieur, et les constantes de temps ne se resserrent que de 41 % par
rapport à leur prior. À constantes de temps *fixées*, la colinéarité paraît
modérée — le conditionnement de la matrice des moindres carrés vaut 5 — mais
dès que τ est échantillonné, τ et k se compensent et la dégénérescence
apparaît. Seul le postérieur la révèle. Le signe de k₂ n'est pas contraint
dans le modèle : le forcer masquerait précisément ce résultat.

**Treize athlètes pour la cible vélo, un seul pour la cible course.** La
réplication ne porte que sur la puissance en vélo, parce qu'elle est la seule
grandeur comparable disponible chez assez de sujets du cache. Tout ce que le
dépôt affirme sur la *course* repose encore sur un athlète, et cet athlète se
trouve être l'un des quatre répondeurs de la cohorte — un cas favorable, choisi
pour d'autres raisons.

**Le planificateur ne connaît pas le futur de la charge.** Il propose une
semaine sous contraintes ; il ne prévoit pas ce qu'elle produira, et le
verdict de dose-réponse qu'il affiche est mesuré sur le passé de l'athlète.
Rien ne garantit qu'un répondeur le reste.

**Qui répond reste inexpliqué.** Vingt-trois athlètes sur 61 montrent un effet
dont l'intervalle exclut zéro, trente-huit non, et cinq le montrent en sens
inverse. Ni la charge moyenne, ni sa variabilité, ni la proportion de jours de
repos, ni la dispersion de la puissance ne séparent les groupes. C'est la
question ouverte du projet, et elle a survécu au passage de 13 à 61 sujets.

**L'enveloppe est autocorrélée par construction**, donc facile à prédire. Le
+0,318 de la persistance mesure l'inertie, pas une compréhension de la
physiologie — et le plancher de détectabilité chiffre ce que cela coûte : il
faudrait un effet de charge de 2 σ pour la battre de façon démontrable sur la
cible course.

**Le confondant de transition de sport reste ouvert.** L'athlète passe du vélo
à la course en 2018, et toute la fenêtre d'évaluation de la cible course est
cette transition : une progression due à l'apprentissage moteur y est
indiscernable d'une progression due à la charge. Une tendance temporelle mise
en concurrence n'absorbe rien (−0,268), ce qui écarte la forme la plus simple
du confondant, pas le confondant lui-même.

**La cible vélo n'est pas un effort maximal déclaré.** `cp20_w` est la
meilleure puissance de vingt minutes d'une sortie, sans le garde-fou
cardiaque que `performance.py` applique à la course. Une enveloppe glissante
prend un maximum, donc une sortie facile ne peut pas abaisser la cible — mais
rien ne garantit que les 417 observations soient des tests, et le +0,279 doit
se lire avec cette réserve.

**Les modèles de fondation reçoivent la charge future.** C'est délibéré — la
question posée est « la charge porte-t-elle de l'information sur la
performance ? », et la lui refuser rendrait un résultat nul inexploitable.
Mais cela veut dire que le +0,657 de Chronos-2 n'est pas un chiffre de
planificateur : en usage réel, la charge à venir n'est pas connue. Le score
honnête d'un planificateur serait plus bas, et il reste à mesurer.

**Un résultat nul ne vaut que sur ces données.** Dix-huit plis, un athlète,
une discipline majoritaire. L'intervalle apparié de [−0,121 · +0,430] dit que
ce protocole ne peut pas exclure un gain réel de covariable allant jusqu'à
+0,43 — il dit seulement qu'il n'en détecte pas. Les deux lectures comptent.

---

## L'échelle de maturité

Le verdict de dose-réponse exige 180 jours de contexte puis huit plis de
28 jours, soit plus de quatre cents jours d'historique. Avant cela
`assess_dose_response` rend « indéterminable » — exact, et inutile : un
nouvel utilisateur n'entend rien pendant plus d'un an sans savoir ni
pourquoi, ni combien de temps encore.

Les capacités du système n'apparaissent pourtant pas toutes en même temps.
`src/evaluation/readiness.py` rend l'échelle, et pour chaque barreau manquant
ce qui manque, en jours ou en mesures :

| capacité | seuil | imposé par |
|---|---|---|
| charge (fitness, fatigue) | 1 jour ; stabilisée après ~τ₁ = 42 j | `compute_load_curves` |
| borne aiguë/chronique | 28 jours pleins | `optimizer._chronic_daily_load` |
| seuils de fraîcheur propres | 30 points | `load_model.form_thresholds` |
| enveloppe de performance | 3 mesures dans 42 jours | `rolling_envelope` |
| prévision évaluable | 1 pli | `horizon_splits` |
| verdict de dose-réponse | 8 plis | `assess_dose_response` |

**Aucun seuil n'est recopié.** Chacun est celui qu'une fonction impose
vraiment, et les comptages passent par ces fonctions — `horizon_splits` pour
les plis, `rolling_envelope` pour l'enveloppe. Un seuil recopié dériverait le
jour où la fonction change, et l'échelle deviendrait un mensonge rassurant.
Un test échoue si l'un des deux bouge.

### Ce que le système peut dire au premier jour

Rien sur l'athlète — mais quelque chose de mesuré sur la population :

```
$ python -m src.cli state      # extrait, athlète au dixième jour

Maturite : 1/6 capacites disponibles

  [oui] charge (fitness, fatigue)          10 jour(s) de calendrier, pas encore stabilisee (t1 = 42 j)
  [non] borne aigu/chronique               exige 28 jours pleins de charge chronique
                                          manque : 18 jour(s) de calendrier
  [non] seuils de fraicheur propres        exige 30 points de fraicheur
                                          manque : 20 jour(s) de calendrier
  [non] enveloppe de performance           0 mesure(s)
                                          manque : 3 mesure(s) dans une fenetre de 42 jours
  [non] prevision evaluable                0 pli(s) d'horizon 28 j
                                          manque : 208 jour(s) d'enveloppe definie
  [non] verdict de dose-reponse            exige 8 plis, 0 disponible(s)
                                          manque : 224 jour(s) (8 pli(s))

  En attendant son propre verdict :
    Chez 23 athlete(s) sur 61 de la cohorte, la charge explique la performance avec un intervalle qui exclut zero ;
    chez 38, non. Rien dans les donnees disponibles a l'inscription ne dit lequel vous etes.
```

Ce prior se lit dans `cohort_replication.parquet`, jamais recopié : un chiffre
de population en dur devient faux dès que la cohorte grandit, et personne ne
s'en aperçoit. Il n'est servi que tant que l'athlète n'a pas son propre
verdict — le lui donner ensuite brouillerait ce qui le concerne.

---

## Garde-fous du dépôt

Ces règles sont appliquées par le code et par les tests, pas seulement
documentées. Le détail est dans [`AGENTS.md`](../AGENTS.md).

- **Provenance obligatoire.** `DataSource` n'a pas de valeur par défaut ;
  écrire du synthétique exige `allow_synthetic=True`. Le tableau de bord a
  autrefois affiché 84 séances générées sous l'étiquette `fit_file`.
- **Aucune métrique ne compte des lignes.** Tout passe par `timebase.py` et
  s'exprime en secondes réelles ; les séries ne sont pas garanties à 1 Hz.
- **Aucune constante physiologique dans le code.** Elles vivent dans
  `config/params.yaml`, et valent `null` tant que la calibration n'a pas
  tourné — une erreur explicite plutôt qu'un nombre inventé.
- **Aucune amélioration affirmée sans validation hors échantillon**, ni sans
  intervalle quand deux variantes se tiennent à quelques millièmes de R². Un
  classement n'est pas une mesure : `report.paired_table` compare les
  variantes point à point, par bootstrap apparié sur les plis.
- **Un modèle qui n'a rien exécuté ne publie rien.** `cross_validate` avale
  les exceptions d'un prédicteur ; une variante cassée ressortirait en
  `folds = 0` avec des NaN, ce qui ressemble à un résultat mesuré.
  `cli evaluate` sort en erreur dans ce cas, et avertit quand deux variantes
  n'ont pas couvert le même nombre de plis.
- **Aucune capacité annoncée n'est crue sur parole.** L'« intégration de
  variables exogènes » de TTM est inerte en zero-shot ; un test le mesure en
  remplaçant les covariables et en vérifiant que la sortie ne bouge pas.
- **Un résultat nul s'accompagne de son plancher.** Affirmer l'absence d'effet
  sans dire quel effet le protocole saurait voir, c'est confondre « on n'a
  rien trouvé » avec « il n'y a rien ». `sensitivity.py` injecte un effet
  d'amplitude connue dans l'enveloppe réelle et rend le plus petit que
  l'intervalle apparié distingue du hasard.
- **Une charge ne s'oppose qu'à une cible de sa discipline.** Opposer une
  charge aux deux tiers cycliste à une vitesse en course a produit le résultat
  nul le plus cité de ce dépôt. Les variantes portent désormais leur
  discipline dans leur nom, et la charge de course sert de contrôle négatif
  sur la cible vélo : si elle y prédisait quoi que ce soit, il faudrait
  chercher une fuite avant de parler de transfert.
- **Un résultat sur un athlète est une anecdote.** `cli cohort` rejoue
  l'expérience sur le cache et rend la **distribution** des résultats, jamais
  leur moyenne : combien d'athlètes voient l'effet, chez combien l'intervalle
  exclut zéro, et dans quel sens se trompent les autres. C'est ce qui a montré
  que l'effet mesuré chez l'athlète de référence ne se généralise pas.
- **Un plan dit ce qu'il n'optimise pas.** Le planificateur cherche la charge
  minimale qui délivre le stimulus, jamais la maximale tolérée : sous une
  dose-réponse inconnue, le coût d'une charge inutile est réel là où son
  bénéfice ne l'est pas. Et `WeeklyPlan.notes` publie ce que le solveur a dû
  abandonner — un plan dégradé qui ne s'annonce pas se lit comme un plan
  optimal.
- **Une borne de sécurité se mesure aussi.** `cli validate-plan` confronte les
  bornes du planificateur à 6768 semaines réelles de 66 athlètes. Le plancher
  de retention en sort **soutenu**, le plafond d'ACWR et celui des jours
  consécutifs **indéterminables** à 0,20 σ près. Toute borne est jugée sur
  deux statistiques — le meilleur effort et l'effort **médian**, insensible au
  nombre de tirages — et sur deux régimes, dont un qui exige un entraînement
  régulier avant la semaine jugée. Sans ces quatre lignes, le protocole naïf
  annonçait que respecter le plafond d'ACWR coûtait 0,333 σ à p < 0,0001 :
  un artefact d'échantillonnage additionné d'un biais de sélection.
- **Un cas isolé se juge contre un taux de base.** Les cinq athlètes chez qui
  la charge dégrade la prévision ont été instruits un par un, et chaque
  statistique d'artefact comparée à sa distribution chez les 23 athlètes
  détectés. Sans groupe de comparaison, n'importe quelle valeur a l'air
  suspecte — quatre des cinq restent dans l'étendue des détectés.
- **Un prédicteur cherché parmi douze se teste hors échantillon.**
  `cli responders` retire un athlète et demande aux autres de le classer,
  prétraitement recalculé dans chaque pli. Aucune famille ne dépasse la classe
  majoritaire — et les deux nuls ne se valent pas : celui de ce qu'on sait au
  premier jour est bien alimenté (plancher 0,05 σ), celui de l'historique ne
  l'est pas (plancher non atteint à 0,40 σ). Un prédicteur **constant** est
  écarté et rapporté comme écarté : le p permuté d'un NaN valait 0,0000, et ce
  prédicteur dégénéré était le seul déclaré retenu de la première exécution.
- **Un paramètre se conteste en bougeant le protocole.** Les constantes de
  temps partagées sortent serrées et incompatibles avec la littérature ;
  l'hypothèse était qu'elles apprennent la fenêtre de lissage de l'enveloppe.
  `cli pool --fenetre 21` la met à l'épreuve, et deux garde-fous la rendent
  lisible. D'abord un **contrôle de composition** : `min_periods` s'applique à
  la fenêtre, donc la raccourcir écarte deux athlètes, et `--exclure` rejoue
  l'ajustement à fenêtre standard sur la cohorte amputée — sans lui, le
  déplacement de τ aurait deux causes mêlées. Ensuite la **caractérisation de
  la cible**, que `describe` publie à chaque ajustement : son autocorrélation
  de rang 1 tombe de +0,539 à +0,343, et sans ce chiffre on attribuerait à la
  physiologie une contraction de **tous** les paramètres de 8 à 19 %. Verdict :
  élasticité 0,22 pour τ₁ et 0,30 pour τ₂ — ni physiologie pure, ni pur
  apprentissage de la fenêtre. Un réglage non publié écrit des artefacts
  suffixés, parce qu'une expérience qui met en doute une mesure ne doit pas
  pouvoir l'écraser.
- **Les tests n'écrivent jamais dans `data/`** ; une fixture le vérifie après
  chaque test. Le module de sensibilité ne déroge pas : l'enveloppe perturbée
  n'existe qu'en mémoire et l'interface ne rend que des statistiques, jamais
  une série qu'on pourrait confondre plus tard avec une mesure.

Le noyau de convolution du modèle bayésien est testé contre le filtre
récursif de `load_model.py` à 1e-10 près : une seule source de vérité pour la
dynamique de charge. Et un test de **récupération de paramètres** simule
depuis des valeurs connues puis vérifie que le postérieur les retrouve — les
cinq tombent dans leur intervalle de crédibilité à 94 %.
