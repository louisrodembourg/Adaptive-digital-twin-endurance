# Digital Twin — Optimisation Semi-Marathon
## Dossier de présentation technique

**Objectif sportif :** 1h22 → 1h18 sur semi-marathon (21.1 km, allure cible 3'40"/km)
**Défi technique :** Système d'IA adaptatif sur données rares (N=1 athlète)
**Stack :** Python · PyMC · OR-Tools · Streamlit · Claude API
**État :** MVP complet — 4 couches opérationnelles, 31 tests, dashboard live

---

## 1. Vue d'Ensemble — Pourquoi ce projet ?

### Le problème réel

Passer de 1h22 à 1h18 n'est pas seulement une question de volume d'entraînement.
C'est un aussi problème d'optimisation sous contraintes multiples :

- **Physiologiques** : fatigue cumulée, récupération du système nerveux autonome, seuil aérobie
- **Mécaniques** : efficacité à l'allure spécifique (3'40"/km), durabilité sur la distance
- **Contextuelles** : charge de travail, sommeil, stress — qui varient chaque jour
- **Incertitude** : avec un seul athlète, pas de "loi des grands nombres" pour compenser le bruit

### La réponse architecturale

Un système cyber-physique en 4 couches, inspiré des systèmes de contrôle industriels :

```
Capteur (montre Coros)
    ↓ données brutes
LAYER A — Signal Lab       : nettoyage + extraction de biomarqueurs
    ↓ métriques fiables
LAYER B — Digital Twin     : modèle d'état probabiliste (où en est l'athlète ?)
    ↓ état + incertitude
LAYER C — Planner          : décision optimale (quelle séance demain ?)
    ↓ plan validé
UI Layer — Dashboard       : monitoring + validation humaine
```

---

## 2. Layer A — Signal Lab

### Responsabilité
Transformer les données brutes de la montre GPS en biomarqueurs fiables et exploitables.

### Le problème des capteurs optiques
Les montres GPS enregistrent la fréquence cardiaque via photopléthysmographie (lumière verte).
Résultat : artéfacts fréquents — spikes, sauts de 56 bpm en une seconde, données manquantes.
**Sur le fichier `longue.fit` (16 km) : 45 sauts anormaux détectés avant nettoyage.**

### Choix technique : Savitzky-Golay plutôt que filtre médian

| Critère | Médian | **Savitzky-Golay (choisi)** |
|---|---|---|
| Déphasage temporel | Oui — décale le signal | **Non** — conserve les transitions |
| Préservation des gradients | Non | **Oui** — crucial pour FC/vitesse |
| Standard industrie | Usage général | **GoldenCheetah, TrainingPeaks** |
| Résultat sur données | Distorsion EF | **EF stable à ±0.1%** |

Le déphasage aurait faussé la corrélation FC ↔ vitesse, donc l'EF. Rédhibitoire.

### Biomarqueurs calculés

**Efficiency Factor Spécifique (EF_spec)** — *La métrique clé du projet*
```
EF_spec = Vitesse_GAP (m/s) / FC (bpm)  — uniquement sur les segments à 3'40"/km ±5%
```
Si EF_spec augmente semaine après semaine → l'athlète devient plus économique à l'allure cible.
C'est le signal direct de progression vers 1h18.

**Grade Adjusted Pace (GAP)** — Correction dénivelé (formule Jack Daniels)
```
speed_GAP = speed × (1 + grade × 0.033)  [montée]
speed_GAP = speed × (1 + grade × 0.018)  [descente]
```
Sans ça, l'EF_spec est biaisé sur terrain vallonné. Même plat apparent = 2-3% de dénivelé total.

**Cardiac Decoupling** — Indicateur de durabilité
```
Decoupling (%) = (EF_1ère_moitié - EF_2ème_moitié) / EF_1ère_moitié × 100
```
- < 5% : bonne endurance aérobie
- > 5% : la FC dérive = le cardio lâche avant les jambes
- > 10% : alerte → annule la séance VMA du lendemain (Safety Valve)

Warm-up 3 minutes exclu du calcul — sans ça, la montée en température cardiaque naturelle
fausse le découplage vers le haut.

**DFA α₁** — Proxy non-invasif du seuil lactatique
```
α₁ < 0.75 → zone aérobie (équivalent seuil lactate ~2 mmol/L)
α₁ > 1.0  → zone anaérobie
```
Implémentation custom de l'analyse DFA (Detrended Fluctuation Analysis) depuis les intervalles RR.
Équivalent d'un test lactate en laboratoire, sans prise de sang.
En attendant les vraies données RR de la montre : génération de RR synthétiques plausibles
via un processus AR(1) calibré sur la variabilité physiologique (RMSSD ~35ms).

**TRIMP (Training Impulse)** — Charge interne normalisée
```
TRIMP = Σ (1/60) × ΔHR_ratio × 0.64 × e^(1.92 × ΔHR_ratio)
```
Modèle exponentiel de Banister : une minute à 95% FCmax vaut bien plus qu'une minute à 70%.
Permet de comparer des séances de types complètement différents sur une échelle unique.

---

## 3. Layer B — Digital Twin (Modèle d'État Bayésien)

### Responsabilité
Estimer l'état invisible de l'athlète — Fitness, Fatigue, forme du jour — avec quantification
de l'incertitude. Le modèle "avoue" quand il ne sait pas.

### Le problème du Small Data

Avec un seul athlète, on ne peut pas utiliser du Machine Learning classique.
Pas assez de données pour entraîner un réseau de neurones. Solution : **mathématiques intelligentes**.

### Choix technique : Banister étendu dans un cadre bayésien (PyMC)

**Modèle Banister (1975)** — équations de Fitness/Fatigue :
```
Fitness(t)     = Σ TRIMP(i) × exp(-(t-i)/τ₁)    τ₁ = 42 jours
Fatigue(t)     = Σ TRIMP(i) × exp(-(t-i)/τ₂)    τ₂ = 12 jours
Performance(t) = p₀ + k₁ × Fitness(t) - k₂ × Fatigue(t)
TSB(t)         = Fitness(t) - Fatigue(t)          [Training Stress Balance]
```

Pourquoi Banister ? Validé dans la littérature sportive depuis 50 ans (Busso 2003, Clarke 2013),
paramètres interprétables, et les constantes de temps ont un sens physiologique réel.

**Extension non-linéaire** (ajout personnel) :
- TSB < -30 → pénalité exponentielle (sur-entraînement)
- TSB > 25 → plateau (rendements décroissants)

Le modèle classique est linéaire — il ne sait pas qu'un athlète épuisé ne progresse plus.

**Cadre bayésien (PyMC + MCMC NUTS)** :

| Approche | Données nécessaires | Incertitude | Interprétabilité |
|---|---|---|---|
| Deep Learning | Milliers d'athlètes | Boîte noire | Faible |
| Banister déterministe | Peu | Aucune | Bonne |
| **Banister Bayésien (choisi)** | **Peu** | **Quantifiée (HDI 95%)** | **Excellente** |

Les priors sont informés par la littérature :
- `τ₁ ~ Normal(42, 5)` — constante fitness : entre 37 et 47 jours selon les études
- `τ₂ ~ Normal(12, 2)` — constante fatigue : 10-14 jours
- `k2 > k1` par construction — la fatigue monte plus vite qu'elle ne descend

**Output du modèle** :
- Courbes Fitness/Fatigue/TSB avec **intervalle de confiance 95% (HDI)**
- `P(Réussir 1h18)` = probabilité calculée depuis la distribution posterior
- Si le cône d'incertitude dépasse un seuil → déclenche automatiquement une séance de calibration

### Gestionnaire d'incertitude

```python
if HDI_width(Fitness) > 15:  # seuil empirique
    → imposer séance test 5km à 4'00"/km
    → la nouvelle observation réduit la posterior
```

Ce mécanisme est ce qui rend le système "vivant" — il reconnaît ses propres limites
et génère des actions pour se recalibrer.

---

## 4. Layer C — Neuro-Symbolic Planner

### Responsabilité
Décider quelle séance faire demain (et les 6 jours suivants) en tenant compte
de l'état physiologique, du contexte personnel, et de l'objectif à long terme.

### Approche Neuro-Symbolique — pourquoi pas juste un LLM ?

Un LLM seul pour planifier l'entraînement présente deux problèmes critiques :
1. **Hallucinations** : peut suggérer une séance VMA à quelqu'un en sur-entraînement
2. **Non-reproductible** : deux appels identiques peuvent donner des plans différents

Solution : **architecture hybride**

```
[LLM — Claude]           [OR-Tools CP-SAT]
"Je suis stressé"   →    max_volume = 0.80
"Mauvais sommeil"   →    no_VMA = True
                         →  Solver déterministe
                            → Plan optimal garanti
```

Le LLM fait ce qu'il fait bien (comprendre le langage naturel).
OR-Tools fait ce qu'il fait bien (optimisation combinatoire exacte).

### Safety Valve — les 6 contraintes dures

Vérifiées AVANT le solver. Non-négociables.

| Signal | Seuil | Action |
|---|---|---|
| rMSSD < 50ms | Système nerveux autonome fatigué | Bloque VMA + Threshold + Specific |
| Découplage hier > 10% | Endurance insuffisante | Repos forcé |
| TSB < -30 | Sur-entraînement fonctionnel | Volume -30% |
| DFA α₁ bas | Intensité réelle trop élevée | Bloque VMA |
| Incertitude modèle élevée | Modèle ne sait plus où il en est | Séance calibration imposée |
| J ≤ 14 avant course | Mode tapering | Volume -20/-40/-70% selon proximité |

### LLM Context Parser

```python
# Input  : "je suis épuisé et j'ai un grand week-end famille"
# Output : {"max_volume_pct": 0.65, "no_session_types": ["VMA","THRESHOLD"],
#           "max_sessions_this_week": 3, "reason": "épuisement + logistique"}
```

Fonctionne en deux modes :
- **Mode Claude API** : compréhension fine du contexte, nuances, ironie
- **Mode heuristique** (fallback) : 10+ règles keyword → opérationnel sans API key

### Optimizer OR-Tools CP-SAT

Problème d'optimisation combinatoire : assigner un type de séance à chacun des 7 jours.

**Variables** : `x[jour][type_session]` — booléen (7 × 8 = 56 variables)

**Objectif** : maximiser `Σ TRIMP × spécificité` — favorise les séances à 3'40"/km

**Contraintes encodées** :
- Max 2 séances haute intensité par semaine (récupération insuffisante au-delà)
- Pas 2 jours intenses consécutifs
- Règle des 10% de progression (TRIMP total ≤ semaine précédente × 1.10)
- Minimum 1 jour de repos
- Sessions bloquées par la Safety Valve

Pourquoi OR-Tools et pas un algorithme greedy ?
L'optimum global n'est pas toujours intuitif. Exemple : mettre la séance SPECIFIC le samedi
au lieu du mardi permet une meilleure récupération post-VMA du jeudi, augmentant la qualité
globale de la semaine. OR-Tools trouve ça en 5ms. Un humain aurait du mal à le voir.

### Module Tapering (J-14)

```
J-14 à J-8  : volume -20%, 1 séance SPECIFIC maintenue
J-7 à J-4   : volume -40%, séances légères + 1 SPECIFIC finale
J-3 à J-1   : RECOVERY uniquement
J            : RACE DAY
```

Maximiser le TSB (fraîcheur) sans perdre la spécificité à l'allure de course.

---

## 5. UI Layer — Dashboard Streamlit

4 pages opérationnelles (`streamlit run app/main_dashboard.py`) :

| Page | Contenu clé |
|---|---|
| **Overview** | Jauge P(1h18), courbes Fitness/Fatigue/TSB interactives, temps estimé |
| **Plan Semaine** | Formulaire contexte matin (rMSSD, sommeil, notes), plan 7 jours coloré, explication XAI |
| **Analyse Séance** | Multi-signaux (FC/vitesse/puissance), zones FC, rapport nettoyage |
| **Tendances** | Progression EF_spec, charge TRIMP, historique rMSSD |

### XAI — Explainability intégrée

Chaque choix du planner est justifié en langage naturel :
> *"VMA annulée mardi : rMSSD=42ms < seuil 50ms. Volume réduit de 25% : épuisement détecté (LLM). Séance SPECIFIC maintenue samedi : priorité spécificité 1h18."*

L'athlète comprend pourquoi le système a décidé ce qu'il a décidé. Il peut overrider en connaissance de cause.

---

## 6. Boucles de Rétroaction — Ce qui rend le système "Intelligent"

### Boucle de Calibration (long terme — hebdomadaire)
```
Modèle prédit une FC pour une séance donnée
    → Réalité s'éloigne de la prédiction (hors HDI)
        → Modèle s'auto-corrige : recalibrage des τ₁, τ₂ sur les nouvelles données
```

### Boucle de Sécurité (court terme — quotidien)
```
Signal Lab détecte anomalie (DFA α₁ bas, découplage élevé)
    → Hard Constraint activée dans Safety Valve
        → OR-Tools ne peut plus assigner VMA/SPECIFIC ce jour
            → Séance remplacée automatiquement par récupération
```

### Boucle de Spécificité (moyen terme — mensuel)
```
EF_spec stagne pendant 3 semaines
    → Optimizer détecte : spécificité trop faible dans le plan
        → Rééquilibrage : +1 séance SPECIFIC/semaine au lieu d'endurance pure
```

---

## 7. Stack Technique Complète

| Composant | Technologie | Justification du choix |
|---|---|---|
| Parsing .fit | `garmin-fit-sdk` | Standard industriel, supporte Coros et Garmin |
| Data pipeline | `pandas` + `parquet` | Columnar pour séries temporelles, compression native |
| Signal cleaning | `scipy.signal.savgol_filter` | Pas de déphasage, standard GoldenCheetah |
| Bayesian model | `PyMC 5` + `PyTensor` | Probabilistic programming, MCMC NUTS, HDI intégré |
| DFA α₁ | Implémentation custom | Indépendance des dépendances, contrôle total |
| Constraint solver | `OR-Tools CP-SAT` | Google Research Operations, exact et rapide (< 5ms) |
| LLM | `Anthropic Claude API` | Meilleur rapport instruction-following / JSON fiable |
| Dashboard | `Streamlit` + `Plotly` | Prototypage rapide, interactivité native |
| Config | `params.yaml` | Aucune constante dans le code, recalibration sans redéploiement |

---


## 8. Ce que ce projet démontre techniquement

**Maîtrise du Small Data**
Pas de fuite vers le Big Data. Le problème est traité avec les bonnes mathématiques :
inférence bayésienne pour l'incertitude, modèle physique pour la structure,
contraintes dures pour la sécurité. C'est plus rigoureux que du ML sur 10 000 athlètes
sans compréhension du domaine.

**Compréhension du domaine métier**
Les métriques implémentées (DFA α₁, découplage cardiaque, EF spécifique, GAP)
sont utilisées par les physiologistes du sport et les plateformes comme TrainingPeaks.
Le code reflète une vraie compréhension de ce qu'on mesure et pourquoi.

**Architecture hybride IA/Symbolique**
Le LLM ne "décide" pas. Il traduit. La décision appartient au solver déterministe,
avec des garanties prouvables. C'est la différence entre un système qui semble intelligent
et un système qui l'est.

**Incertitude explicite**
Le modèle bayésien n'est pas un oracle — il donne une distribution, pas un point.
Quand il ne sait pas, il le dit et déclenche une action pour apprendre davantage.
C'est ce qui distingue un modèle de production d'un modèle de démo.

**Ingénierie logicielle**
Configuration centralisée (YAML), séparation claire des couches, fallbacks gracieux
(PyMC absent → mode déterministe, OR-Tools absent → greedy, API indisponible → heuristique),
tests end-to-end, persistance Parquet.

---

## 10. Prochaines Étapes

| Priorité | Action | Impact |
|---|---|---|
| 🔴 Critique | Activer enregistrement RR sur Coros (ou HRV Logger) | Débloquer le vrai DFA α₁ temps réel |
| 🟠 Important | Entrer les performances passées réelles dans le modèle PyMC | Calibrer la posterior sur l'historique réel |
| 🟡 Utile | Normaliser k₁/k₂ (Fitness/Fatigue en unités relatives) | Valeurs TSB interprétables directement |
| 🟡 Utile | CI/CD — test automatique à chaque import de nouveau .fit | Régression impossible en production |
| 🟢 Extension | Multi-athlètes avec priors partagés (hierarchical Bayes) | Passage à l'échelle coaching |

---

*Projet personnel — Louis Rodembourg — 2026*
*Code disponible sur demande*
