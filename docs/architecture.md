# Architecture and getting started

The modules, the data flow, and the commands.

Short presentation: [README](../README.md).

---

## Architecture

```mermaid
flowchart TB
    OSF[("GoldenCheetah OpenData<br/>6,614 athletes")]

    subgraph A["Layer A — Signal"]
        direction LR
        ING["opendata.py<br/><i>cache-first, md5</i>"]
        TB["timebase.py<br/><i>real seconds</i>"]
        FEAT["features.py<br/><i>TRIMP · EF · GAP</i>"]
        PERF["performance.py<br/><i>Riegel</i>"]
        ING --> TB --> FEAT --> PERF
    end

    subgraph P["Pipeline"]
        direction LR
        PROV["provenance.py<br/><i>no invented data<br/>can pass as measured</i>"]
        ORCH["orchestrator.py<br/><i>ingest → clean → featurize</i>"]
        PROV --> ORCH
    end

    subgraph STORE["Artifacts"]
        direction LR
        S1[("sessions")]
        S2[("daily_load<br/><i>run · bike · other</i>")]
        S3[("perf_observations<br/><i>+ per-duration index</i>")]
    end

    ENV["envelope.py<br/><i>rolling envelope<br/>daily or observation grid</i>"]

    subgraph B["Layer B — Twin"]
        direction LR
        LOAD["load_model.py<br/><i>CTL · ATL · freshness<br/>per-discipline curves</i>"]
        BAYES["bayesian_model.py<br/><i>PyMC · convolution</i>"]
        HIER["hierarchical.py<br/><i>pooling · 62 athletes</i>"]
        LOAD --> BAYES --> HIER
    end

    subgraph E["Evaluation"]
        direction LR
        BASE["baselines.py<br/><i>declared in advance</i>"]
        FND["foundation.py<br/><i>Chronos-2 · TTM<br/>covariate ablation</i>"]
        SENS["sensitivity.py<br/><i>detectability floor</i>"]
        COH["cohort.py<br/><i>replication · 61 athletes</i>"]
        VALID["validation.py<br/><i>blocks · 28-day horizon · per duration</i>"]
        REPORT["report.py<br/><i>paired block bootstrap</i>"]
        BASE --> VALID --> REPORT
        FND --> VALID
        RULES["plan_rules.py<br/><i>plan bounds · 6,768 weeks</i>"]
        RESP["responders.py<br/><i>who responds · out of sample</i>"]
        READY["readiness.py<br/><i>maturity ladder</i>"]
        COH --> RESP
        COH --> READY
        REPORT --> SENS
        REPORT --> COH
    end

    subgraph C["Layer C — Planner"]
        direction LR
        DOSE["dose_response.py<br/><i>per-athlete verdict</i>"]
        SAFE["safety_valve.py<br/><i>ACWR · HRV · taper</i>"]
        LLM["llm_context_parser.py<br/><i>free-text constraints</i>"]
        OPT["optimizer.py<br/><i>CP-SAT · minimum<br/>sufficient load</i>"]
        SAFE --> OPT
        DOSE --> OPT
        LLM --> OPT
        RULES -.->|"validates its bounds"| SAFE
    end

    UI["app/main_dashboard.py<br/><i>Streamlit · 5 pages</i>"]

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
    class REPORT,BAYES,FND,SENS,COH,HIER,RULES,RESP,READY result
    class PROV,TB guard
```

| layer | role |
|---|---|
| **A — Signal** | raw sessions → cleaned series on a real-seconds timebase → TRIMP, efficiency, grade-adjusted pace, performance observations |
| **Pipeline** | orchestration and provenance: every session carries an explicit `DataSource` |
| **B — Twin** | fitness-fatigue load curves per discipline, single-athlete Bayesian model, hierarchical pooling across athletes |
| **Evaluation** | declared baselines, time-ordered validation, paired block bootstrap, detectability floors, cohort replication, safety-rule validation, responder prediction, maturity ladder |
| **C — Planner** | per-athlete dose-response verdict, safety bounds, constraints parsed from free text, CP-SAT optimiser looking for the minimum sufficient load |

---

## Getting started

```bash
pip install -e ".[dev]"

python -m src.cli fetch-athletes --critere course --keep 3   # find a subject
python -m src.cli ingest                    # build the three artifacts
python -m src.cli calibrate                 # write estimates to params.yaml
python -m src.cli state                     # summarise what exists
python -m src.cli evaluate                  # five protocols + floors
python -m src.cli cohort                    # replicate on the athlete cache
python -m src.cli plan                      # propose a week, and what it is worth
python -m src.cli validate-plan             # confront its bounds with data
python -m src.cli responders                # who responds, out of sample
python -m src.cli pool                      # hierarchical pooling (~2 h 30)
python -m src.cli pool --decrire            # the target only, no sampling

streamlit run app/main_dashboard.py
pytest                                      # offline by default
```

Foundation models are optional, because `torch` weighs about 2 GB and no
number in the main results needs it to be reproduced:

```bash
pip install -e ".[tsfm]"
python -m src.cli evaluate --models chronos2,ttm
```

Without `data/processed`, the dashboard reads the frozen snapshot in
[`demo/processed`](../demo/README.md) — real artifacts from the public dataset —
and says so in its sidebar. That is what the online demo serves.

These commands reproduce **every published number** from a public dataset.
That is what makes them verifiable rather than declarative.

The command-line interface and the dashboard speak French; flags such as
`--critere`, `--fenetre`, `--exclure` and `--decrire` are the French words for
criterion, window, exclude and describe.

---

## Stack

`pandas` · `numpy` · `scipy` · `pyarrow` · `PyMC` / `ArviZ` · `OR-Tools`
CP-SAT · `Streamlit` · `Plotly` · `pytest` · `ruff`

Optional, for the foundation-model baselines: `torch` ·
`chronos-forecasting` · `granite-tsfm`.

On the evaluation side, five protocols come out of a single command:
positional blocks, calendar horizon, per-duration envelope, bike target, and
detectability floor.

Data: [GoldenCheetah OpenData](https://osf.io/6hfpz/), DOI
[10.17605/OSF.IO/6HFPZ](https://doi.org/10.17605/OSF.IO/6HFPZ).

---

*Personal project — Louis Rodembourg*
