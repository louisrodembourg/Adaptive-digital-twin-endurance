# Architecture et mise en route

Les modules, le flux de données, et les commandes.

Présentation courte : [README](../README.md).

---

## Architecture

```mermaid
flowchart TB
    OSF[("GoldenCheetah OpenData<br/>6 614 athlètes")]

    subgraph A["Layer A — Signal"]
        direction LR
        ING["opendata.py<br/><i>cache-first, md5</i>"]
        TB["timebase.py<br/><i>secondes réelles</i>"]
        FEAT["features.py<br/><i>TRIMP · EF · GAP</i>"]
        PERF["performance.py<br/><i>Riegel</i>"]
        ING --> TB --> FEAT --> PERF
    end

    subgraph P["Pipeline"]
        direction LR
        PROV["provenance.py<br/><i>aucune donnée inventée<br/>ne peut passer pour mesurée</i>"]
        ORCH["orchestrator.py<br/><i>ingest → clean → featurize</i>"]
        PROV --> ORCH
    end

    subgraph STORE["Artefacts"]
        direction LR
        S1[("sessions")]
        S2[("daily_load<br/><i>run · bike · other</i>")]
        S3[("perf_observations<br/><i>+ index par durée</i>")]
    end

    ENV["envelope.py<br/><i>enveloppe glissante<br/>grille jour ou observation</i>"]

    subgraph B["Layer B — Jumeau"]
        direction LR
        LOAD["load_model.py<br/><i>CTL · ATL · fraîcheur<br/>courbes par discipline</i>"]
        BAYES["bayesian_model.py<br/><i>PyMC · convolution</i>"]
        LOAD --> BAYES
    end

    subgraph E["Évaluation"]
        direction LR
        BASE["baselines.py<br/><i>déclarées d'avance</i>"]
        FND["foundation.py<br/><i>Chronos-2 · TTM<br/>ablation de covariable</i>"]
        SENS["sensitivity.py<br/><i>plancher de détectabilité</i>"]
        COH["cohort.py<br/><i>réplication · 13 athlètes</i>"]
        VALID["validation.py<br/><i>blocs · horizon 28 j · par durée</i>"]
        REPORT["report.py<br/><i>bootstrap apparié par blocs</i>"]
        BASE --> VALID --> REPORT
        FND --> VALID
        REPORT --> SENS
        REPORT --> COH
    end

    subgraph C["Layer C — Planificateur"]
        direction LR
        DOSE["dose_response.py<br/><i>verdict par athlète</i>"]
        SAFE["safety_valve.py<br/><i>ACWR · VFC · taper</i>"]
        OPT["optimizer.py<br/><i>CP-SAT · charge minimale<br/>suffisante</i>"]
        SAFE --> OPT
        DOSE --> OPT
    end

    UI["app/main_dashboard.py<br/><i>Streamlit · 4 pages</i>"]

    OSF --> A --> P --> STORE
    STORE --> ENV --> E
    STORE --> B
    B --> E
    B --> C
    E --> DOSE
    E --> UI
    C --> UI
    STORE --> UI

    classDef result fill:#16324f,stroke:#4c8bf5,stroke-width:2px,color:#fff
    classDef guard fill:#4a3620,stroke:#e0a355,stroke-width:2px,color:#fff
    class REPORT,BAYES,FND,SENS,COH result
    class PROV,TB guard
```

---

## Démarrage

```bash
pip install -e ".[dev]"

python -m src.cli screen-athletes --n 200   # trouver un athlète exploitable
python -m src.cli ingest                    # construire les trois artefacts
python -m src.cli calibrate                 # reporter les estimations
python -m src.cli state                     # résumer ce qui existe
python -m src.cli evaluate                  # cinq protocoles + planchers
python -m src.cli cohort                    # répliquer sur le cache d'athlètes
python -m src.cli plan                      # proposer une semaine, et ce qu'elle vaut

streamlit run app/main_dashboard.py
pytest                                      # hors ligne par défaut
```

Les modèles de fondation sont en option, parce que `torch` pèse environ 2 Go
et qu'aucun chiffre ci-dessus n'en a besoin pour être reproduit :

```bash
pip install -e ".[tsfm]"
python -m src.cli evaluate --models chronos2,ttm
```

Ces commandes reproduisent **chaque chiffre publié** à partir d'un jeu
de données public. C'est ce qui les rend vérifiables plutôt que déclaratives.

---

## Stack

`pandas` · `numpy` · `scipy` · `pyarrow` · `PyMC` / `ArviZ` · `OR-Tools`
CP-SAT · `Streamlit` · `Plotly` · `pytest` · `ruff`

En option, pour les références à modèle de fondation : `torch` ·
`chronos-forecasting` · `granite-tsfm`.

Côté évaluation, cinq protocoles sortent d'une seule commande : blocs
positionnels, horizon calendaire, enveloppe par durée d'effort, cible vélo, et
plancher de détectabilité.

Données : [GoldenCheetah OpenData](https://osf.io/6hfpz/), DOI
[10.17605/OSF.IO/6HFPZ](https://doi.org/10.17605/OSF.IO/6HFPZ).

---

*Projet personnel — Louis Rodembourg*
