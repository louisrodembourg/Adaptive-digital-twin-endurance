# Architecture du Projet DigitalTwin V2

**Objectif Sportif :** Optimisation Semi-Marathon (1h22 $\rightarrow$ 1h18)
**Défi Technique :** Modélisation sur données rares ($N=1$) et hybridation IA/Physiologie.

## 1. Vue d'Ensemble du Flux de Données (Pipeline)

Le système fonctionne en mode hybride pour allier vision à long terme et réactivité physiologique au jour le jour.

### Cycle Hebdomadaire (Batch du Dimanche Soir)
- **Ingestion :** Import des fichiers `.fit` (Coros) de la semaine écoulée.
- **Traitement Lourd :** Nettoyage approfondi des signaux (ex: correction des artefacts cardiaques) et extraction des features avancées (Physio + Mécanique).
- **Modélisation :** Mise à jour du modèle d'état dynamique (Fitness/Fatigue) et calcul du cône d'incertitude via Inférence Bayésienne.
- **Planification Globale :** Génération du plan d'entraînement de base pour la semaine suivante par le Solver.

### Cycle Quotidien (Matin / Temps Réel)
- **Ingestion Rapide :** Récupération du contexte journalier (HRV/rMSSD, qualité du sommeil, RPE, stress).
- **Safety Valve :** Évaluation de la fraîcheur et vérification des contraintes dures (ex: alerte fatigue aiguë).
- **Ré-Optimisation :** Ajustement dynamique du plan du jour par le Solver (ex: bascule d'une séance VMA vers de l'endurance si le système nerveux est fatigué).

### En Continu
- **Visualisation :** Mise à jour du Dashboard Streamlit pour le monitoring et la validation humaine des choix de l'IA.

---

## 2. Détail des Composants & Stack Technique

### A. Ingestion & Signal Lab (ETL & Feature Engineering)
*Responsabilité : Transformer la donnée brute en biomarqueurs fiables.*
*Localisation : `src/layer_a_signal/`*

- **Stack :** `Python` (`fitparse`, `Pandas`), `NeuroKit2`
- **Fonctionnalités Clés :**
    - **Nettoyage DFA $\alpha_1$ :** **Correction stricte des artefacts** cardiaques sur les intervalles RR (algorithme type Kubios) *avant* le calcul de l'indice fractal pour valider l'intensité réelle. (Seuil Aérobie). =⇒ sert pour avoir un équivalent de lactate
    - **Specific Efficiency Factor ($EF_{spec}$) :** Calcul du ratio $Vitesse_{GAP} / FC$ uniquement sur les segments courus à l'allure cible (3'40"/km). *Crucial pour l'obj 1h18*
    - **Cardiac Decoupling (Durabilité) :** Détection de la dérive cardiaque (>5%) sur les sorties longues (>1h15).
    - **Normalisation de la Charge :** Conversion de la charge Mécanique (Muscu) en équivalent TRIMP pour un score de charge unique.

### B. Le Cœur : Modèle d'État Bayésien (Probabilistic PINN)
*Responsabilité : Estimer l'état invisible de l'athlète (Fitness, Fatigue) avec quantification de l'incertitude.*
*Localisation : `src/layer_b_twin/`*

- **Stack :** `PyMC` (ou `NumPyro`)
- **Architecture :** Modèle State-Space **dynamique (type Busso)** intégrant des paramètres non-linéaires pour gérer l'accumulation de fatigue extrême.
    - **Priors Informés :** Initialisation des paramètres ($\tau_1, \tau_2$) basée sur la littérature et calibrée sur ton record actuel (1h22).
    - **Likelihood :** Mise à jour des courbes en fonction des performances réelles observées chaque semaine.
    - **Output :** Courbes de Fitness avec **Cône d'Incertitude** (Confidence Interval 95%). Le modèle "avoue" quand il ne sait pas.

### C. Le Planificateur Neuro-Symbolique
*Responsabilité : Décision et création de l'emploi du temps.*
*Localisation : `src/layer_c_planner/`*

- **Stack :** `Google OR-Tools` (CP-SAT Solver) + `LLM` (API via script)
- **Approche Hybride :**
    - **LLM (Analyseur de Contexte) :** Traduit *"Je suis stressé cette semaine"* en contrainte mathématique $\rightarrow$ `Max_Vol_Global = 0.8 * Previous_Vol`.
    - **Solver (Calculateur Déterministe) :** Résout le problème d'emploi du temps.
- **Objectif :** Maximiser la charge pondérée.
- **Contraintes Dures :**
    - Pas de VMA si `rMSSD` < Seuil.
    - Règle des 10% de progression.
    - Repos forcé si `Decoupling` > 5% la veille.
    - **Calibration Forcée** : Si le Cône d'Incertitude (Output Bayésien) dépasse un seuil critique X, l'optimiseur annule le travail spécifique et planifie une séance test standard pour réduire l'incertitude.
- **Module Tapering (J-14) :** Algorithme spécifique pour maximiser la fraîcheur (TSB) sans perdre la Fitness spécifique à l'approche du jour J.

### D. Interface & Explainability (XAI)
*Responsabilité : Interaction Utilisateur.*
*Localisation : `app/main_dashboard.py`*

- **Stack :** `Streamlit`
- **Vues :**
    - Jauge de "Probabilité de Réussite 1h18" (basée sur le modèle bayésien).
    - Graphique de l'évolution de l'Efficacité Spécifique ($EF_{spec}$).
    - **Feedback :** *"Plan réduit de 15% car HRV bas et dérive cardiaque détectée dimanche."*

---

## 3. Interactions et Boucles de Rétroaction
*Ce qui rend ton système "Intelligent" (Cyber-Physique).*

- **🔄 Boucle de Calibration (Long terme) :**
Ton modèle Bayésien prédit une FC pour une séance donnée. Si la réalité s'éloigne trop de la prédiction (hors du cône d'incertitude), le modèle s'auto-corrige la semaine suivante en ajustant ses constantes de temps ($\tau$).

- **🛡️ Boucle de Sécurité (Court terme) :**
Le module Signal détecte une anomalie (ex: DFA $\alpha_1$ trop bas sur une sortie facile = fatigue nerveuse). Cette info devient une "Hard Constraint" pour OR-Tools $\rightarrow$ La séance VMA prévue mardi est automatiquement remplacée par un footing ou du repos.

- **🎯 Boucle de Spécificité (Obj 1h18) :**
Si le `Specific Efficiency Factor` à 3'40"/km stagne, l'optimiseur change la distribution des intensités pour privilégier le travail au seuil plutôt que le volume pur.
