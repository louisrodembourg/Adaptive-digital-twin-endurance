# DigitalTwin — TODO Roadmap

> **Objectif : Semi-Marathon 1h18** (actuel 1h22)
> Chaque phase produit un livrable testable. On ne passe à la suivante que quand la précédente est **validée**.

---

## Phase 1 — La Fondation (Ingestion Data)
*Lire et structurer la donnée brute des fichiers .fit Coros.*
*Localisation : `src/layer_a_signal/ingest_fit.py`*

- [x] Parser les fichiers `.fit` avec `garmin-fit-sdk` (robuste aux champs développeurs Coros)
- [x] Extraire les colonnes clés : timestamp, HR, vitesse, puissance, cadence, altitude, running dynamics
- [x] Convertir les unités (m/s → km/h, pace min/km)
- [x] Extraire les métadonnées de session (`get_session_summary()`)
- [ ] Gérer l'import batch (dossier de `.fit` → DataFrame multi-séances)
- [ ] Sauvegarder en Parquet dans `data/processed/`
- [ ] **HRV / Intervalles RR :**
  - [ ] Activer "HRV Logging" sur Coros PACE 4 ou enregistrer via app tierce (HRV Logger)
  - [ ] Parser les RR depuis la source choisie (`.fit` HRV ou `.csv` externe)
  - [ ] Fusionner RR + données 1Hz par timestamp
- [ ] Ingestion contexte quotidien (HRV nocturne rMSSD, sommeil, RPE, stress) → `data/daily_context.csv`

---

## Phase 2 — Le Laboratoire (Signal & Features)
*Nettoyer le signal et coder les métriques clés.*
*Localisation : `src/layer_a_signal/features.py`*

- [x] Nettoyage du signal (`clean_signal()`) : artefacts FC, interpolation, filtrage marche/course
- [x] **Efficiency Factor global** : Vitesse / FC sur les segments de course
- [x] **EF Spécifique ($EF_{spec}$)** : EF filtré à l'allure cible 3'40"/km ± 5%
- [x] **Cardiac Decoupling** : dérive cardiaque 1ère vs 2nde moitié (seuil 5%)
- [x] **TRIMP** (Banister exponentiel) : charge interne par séance
- [x] **Zones FC** : temps passé par zone (Z1–Z5)
- [x] **Normalized Power** : rolling 30s, puissance 4ème
- [x] `session_features()` : calcul de TOUTES les features en un appel
- [ ] **DFA $\alpha_1$** (en attente des RR) :
  - [ ] Correction des artefacts RR (algorithme type Kubios)
  - [ ] Calcul DFA alpha1 par fenêtre glissante (NeuroKit2 ou implémentation maison)
  - [ ] Détection du seuil aérobie (DFA $\alpha_1$ = 0.75)
- [ ] **GAP (Grade Adjusted Pace)** : correction de la vitesse par le dénivelé pour $EF_{spec}$
- [ ] Normalisation charge muscu → équivalent TRIMP
- [ ] Notebook de visualisation exploratoire (`02_feature_engineering.ipynb`)
  - [x] Résumé chiffré de séance
  - [x] Graphiques multi-signaux (FC, Vitesse, Puissance, Cadence)
  - [x] Visualisation Cardiac Decoupling + trend
  - [x] EF par blocs de 5 min
  - [ ] Heatmap zones d'intensité
  - [ ] Comparaison multi-séances (quand batch disponible)

---

## Phase 3 — Le Cœur (Modèle Bayésien)
*Implémenter les équations de Busso dans PyMC. Obtenir les courbes Fitness/Fatigue et le cône d'incertitude.*
*Localisation : `src/layer_b_twin/`*

- [ ] Implémenter le modèle de Busso (Fitness-Fatigue) :
  - [ ] Équations d'état : $g(n) = g(n-1) \cdot e^{-1/\tau_1} + k_1 \cdot w(n)$
  - [ ] Paramètres : $\tau_1$ (fitness ~42j), $\tau_2$ (fatigue ~12j), $k_1$, $k_2$
  - [ ] Performance prédite = $p_0 + \text{Fitness}(n) - \text{Fatigue}(n)$
- [ ] Priors informés depuis la littérature + calibration sur le record 1h22
- [ ] Likelihood : relier la performance observée (TRIMP, vitesse, FC) au modèle
- [ ] Inférence MCMC avec PyMC (ou NumPyro si perf nécessaire)
- [ ] **Output : Courbes Fitness + Fatigue + TSB avec cône d'incertitude 95%**
- [ ] Boucle de calibration : re-fit hebdomadaire avec les nouvelles données
- [ ] Détection automatique quand le cône d'incertitude dépasse un seuil critique
- [ ] Notebook `03_bayesian_model.ipynb` pour validation et visualisation
- [ ] Tests unitaires sur données synthétiques

---

## Phase 4 — Le Cerveau (L'Optimiseur)
*Paramétrer OR-Tools avec les contraintes dures, connecter le LLM pour le contexte.*
*Localisation : `src/layer_c_planner/`*

- [ ] Définir le modèle de programmation par contraintes (CP-SAT) :
  - [ ] Variables : type de séance par jour (repos, endurance, seuil, VMA, SL, muscu)
  - [ ] Objectif : maximiser la charge pondérée sur la semaine
- [ ] **Contraintes dures :**
  - [ ] Pas de VMA si rMSSD < seuil
  - [ ] Règle des 10% de progression hebdomadaire
  - [ ] Repos forcé si Decoupling > 5% la veille
  - [ ] Calibration forcée si cône d'incertitude > seuil X
  - [ ] Alternance obligatoire charge/récup
- [ ] **Contraintes souples :**
  - [ ] Préférences jours/horaires
  - [ ] Disponibilité variable
- [ ] **Module Tapering (J-14) :**
  - [ ] Algorithme de réduction progressive du volume
  - [ ] Maintien de l'intensité spécifique
  - [ ] Maximisation du TSB à J-0
- [ ] **Connexion LLM :**
  - [ ] Prompt engineering : "Je suis stressé" → `Max_Vol = 0.8 * Prev_Vol`
  - [ ] Interface API (OpenAI/Claude) → contraintes mathématiques
  - [ ] Safety Valve quotidienne (veto/ajustement du plan du jour)
- [ ] Notebook `04_planner.ipynb` pour tests et validation
- [ ] Tests : vérifier que le solver respecte toutes les contraintes

---

## Phase 5 — La Vitrine (Dashboard Streamlit)
*Créer le dashboard visuel pour piloter le tout.*
*Localisation : `app/main_dashboard.py`*

- [ ] Page 1 — Vue d'ensemble :
  - [ ] Jauge "Probabilité de Réussite 1h18"
  - [ ] Courbes Fitness/Fatigue/TSB avec cône d'incertitude
  - [ ] Countdown J-X avant la course
- [ ] Page 2 — Analyse de séance :
  - [ ] Multi-signaux interactifs (FC, Vitesse, Puissance)
  - [ ] Cardiac Decoupling visuel
  - [ ] EF spécifique dans le temps
- [ ] Page 3 — Plan de la semaine :
  - [ ] Calendrier avec séances proposées par le Solver
  - [ ] Boutons de validation / override humain
  - [ ] Feedback expliqué : *"Plan réduit de 15% car HRV bas"*
- [ ] Page 4 — Historique & Tendances :
  - [ ] Évolution $EF_{spec}$ semaine par semaine
  - [ ] Évolution TRIMP / charge
  - [ ] Zones d'intensité cumulées
- [ ] Intégration du contexte quotidien (formulaire RPE/stress/sommeil)
- [ ] Mode sombre / responsive

---

## Transversal

- [ ] CI/CD : linting, tests auto, pre-commit hooks
- [ ] Documentation API (docstrings déjà en place)
- [ ] Versioning des modèles (MLflow ou simple JSON)
- [ ] Backup automatique des données `.fit` et `processed/`

---

*Dernière mise à jour : 24 février 2026*
