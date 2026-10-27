# Demo snapshot

The artifacts the online dashboard reads, so that it can run without
downloading and processing the public dataset (~3 GB of archives, hours of
MCMC for the pooled model).

**Real data, frozen.** Every file here was produced by the CLI on
[GoldenCheetah OpenData](https://osf.io/6hfpz/) — public, anonymised, no GPS
coordinates. The 917 sessions all carry the `opendata_osf` source; nothing is
synthetic. The dashboard says it is reading this snapshot in its sidebar.

`data/processed` always takes precedence: when it holds sessions, the dashboard
reads it and ignores this folder. To rebuild the snapshot from scratch:

```bash
python -m src.cli ingest && python -m src.cli calibrate
python -m src.cli evaluate && python -m src.cli cohort
python -m src.cli validate-plan && python -m src.cli responders
python -m src.cli pool            # plus --fenetre 21 and --exclure for the two controls
cp data/processed/{sessions,daily_load,perf_observations}.parquet \
   data/processed/athlete_calibration.json \
   data/processed/evaluation_*.parquet data/processed/evaluation_provenance.json \
   data/processed/cohort_replication.parquet data/processed/plan_rules.parquet \
   data/processed/pooling_summary*.parquet data/processed/pooling_effects.parquet \
   data/processed/responders_correlations.parquet demo/processed/
```
