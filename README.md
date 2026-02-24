# Adaptive Digital Twin - Endurance (V2 & V3 Architecture)

**Application d'Optimisation d'Entraînement Semi-Marathon assistée par IA Hybride.**

Ce projet est un "Jumeau Numérique" pour athlète d'endurance. Il combine **Science des Données**, **Modélisation Bayésienne** et **Optimisation sous contraintes** pour générer des plans d'entraînement adaptatifs et ultra-personnalisés.

## 🎯 Objectif Sportif
*   **Distance :** Semi-Marathon
*   **Objectif Chrono :** Passer de **1h22** à **1h18** (3'40"/km).
*   **Défi Technique :** Modélisation robuste sur peu de données ($N=1$) via l'hybridation IA/Physiologie.

---

## 🏗 Architecture du Système (Pipeline)

Le système fonctionne en mode **Batch Hebdomadaire** :

1.  **Dimanche Soir :** Ingestion des fichiers `.fit` (Coros) et données contextuelles.
2.  **Traitement (Signal Lab) :** Nettoyage, extraction de features avancées (Physio + Mécanique).
3.  **Modélisation (Digital Twin) :** Mise à jour des états de Fitness/Fatigue via Inférence Bayésienne.
4.  **Optimisation (Planner) :** Génération du plan de la semaine suivante sous contraintes.
5.  **Visualisation (UI) :** Dashboard Streamlit pour validation humaine.

---

## 🔧 Composants & Stack Technique

### 1. Ingestion & Signal Lab (ETL)
*Responsabilité : Transformer la donnée brute en biomarqueurs fiables.*
*   **Stack :** `Python` (`fitparse`, `Pandas`), `NeuroKit2`.
*   **Features Clés :**
    *   **DFA $\alpha_1$ :** Analyse fractale de la variabilité cardiaque pour valider l'intensité réelle.
    *   **Specific Efficiency Factor ($EF_{spec}$) :** Ratio $Vitesse_{GAP} / FC$ calculé uniquement à allure cible (3'40"/km).
    *   **Cardiac Decoupling :** Mesure de la durabilité (dérive cardiaque sur sorties longues).
    *   **Normalisation :** Fusion de la charge cardio et mécanique.

### 2. Le Cœur : Modèle Bayésien (Digital Twin)
*Responsabilité : Estimer l'état invisible (Fitness, Fatigue) avec quantification de l'incertitude.*
*   **Stack :** `PyMC` (Probabilistic Programming).
*   **Logique :** Modèle State-Space (Banister) augmenté.
    *   **Priors Informés :** Calibrés sur la physiologie humaine et l'historique athlète.
    *   **Likelihood :** Mise à jour bayésienne basée sur les performances réelles ($EF_{spec}$).
    *   **Output :** Courbes de forme avec **Intervalle de Confiance (95% HDI)**.

### 3. Planificateur Neuro-Symbolique (Planner)
*Responsabilité : Décision et création de l'emploi du temps.*
*   **Stack :** `Google OR-Tools` (CP-SAT Solver) + `LLM` (pour le parsing contextuel).
*   **Logique Hybride :**
    *   **LLM :** Traduit le contexte qualitatif (stress, fatigue ressentie) en contraintes mathématiques.
    *   **Solver :** Résout le problème d'optimisation combinatoire.
*   **Contraintes Dures :**
    *   Si `rMSSD` < Seuil ou `Decoupling` > 5% $\rightarrow$ Pas de haute intensité.
    *   Progression volume ≤ 10% / semaine.
    *   Module spécifique "Tapering" (Affûtage) à J-14.

### 4. Interface & Explicabilité (UI)
*Responsabilité : Interaction Utilisateur et Trust.*
*   **Stack :** `Streamlit`.
*   **Visualisations :**
    *   Jauge "Probabilité de réussite 1h18".
    *   Évolution de l'Efficacité Spécifique.
    *   **XAI (Explainable AI) :** Justification textuelle des décisions du Solver (ex: "Séance annulée cause fatigue nerveuse").

---

## 🧠 Boucles de Rétroaction "Intelligentes"

Ce qui différencie ce système d'un fichier Excel statique :

1.  **🔄 Boucle de Calibration (Long terme) :** Si la réalité s'éloigne des prédictions du modèle (hors du cône d'incertitude), le modèle s'auto-corrige en recalibrant ses constantes de temps ($\tau$).
2.  **🛡️ Boucle de Sécurité (Court terme) :** Une anomalie physiologique détectée (rMSSD bas, Dérive cardiaque) devient instantanément une "Hard Constraint" pour l'optimiseur, modifiant le plan à venir pour éviter le surentraînement.
3.  **🎯 Boucle de Spécificité :** Si l'Efficacité à l'allure cible stagne, l'optimiseur réajuste la distribution d'intensité pour privilégier le travail au seuil spécifique.

---

## 🚀 Installation & Lancement (V1 Prototype)

1.  Cloner le repo :
    ```bash
    git clone https://github.com/louisrodembourg/Adaptive-digital-twin-endurance.git
    cd Adaptive-digital-twin-endurance
    ```
2.  Créer l'environnement virtuel et installer les dépendances :
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  Lancer l'application :
    ```bash
    streamlit run main.py
    ```

---
*Projet développé par Louis Rodembourg.*
