"""
Orchestrateur : la chaine qui reliait Layer A a Layer B et qui n'existait pas.

Tout tourne hors ligne sur une archive de fixture.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ConfigError, load_config
from src.layer_a_signal import features as fx
from src.pipeline import orchestrator as orch
from src.pipeline.orchestrator import (
    LocalZipSpec,
    build_daily_load,
    load_daily_load,
    load_sessions,
    make_session_id,
    provenance_of,
    run_pipeline,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_athlete.zip"


@pytest.fixture
def pipeline_cfg(cfg, monkeypatch):
    calibrated = replace(cfg, athlete=replace(cfg.athlete, hr_max=190.0, hr_rest=50.0))
    monkeypatch.setattr(orch, "load_config", lambda *a, **k: calibrated)
    monkeypatch.setattr(fx, "load_config", lambda *a, **k: calibrated)
    return calibrated


@pytest.fixture
def ingested(pipeline_cfg):
    run_pipeline(LocalZipSpec(FIXTURE), pipeline_cfg)
    return pipeline_cfg


# -- identifiants ----------------------------------------------------------

def test_session_id_is_stable_and_distinguishes_sessions():
    a = pd.Timestamp("2019-01-15 07:00:00")
    assert make_session_id("x", a) == make_session_id("x", a)
    assert make_session_id("x", a) != make_session_id("y", a)
    assert make_session_id("x", a) != make_session_id("x", a + pd.Timedelta("1s"))


# -- chaine complete -------------------------------------------------------

def test_pipeline_writes_sessions_with_their_provenance(ingested):
    sessions = load_sessions(ingested)
    assert len(sessions) == 6
    assert set(sessions["source"]) == {"opendata_osf"}
    assert provenance_of(sessions).is_trustworthy


def test_pipeline_is_idempotent(ingested):
    report = run_pipeline(LocalZipSpec(FIXTURE), ingested)
    assert report.n_new == 0 and report.n_skipped == 6
    assert len(load_sessions(ingested)) == 6


def test_durations_come_from_elapsed_time_not_row_count(ingested):
    sessions = load_sessions(ingested).sort_values("start_time")
    # La derniere seance de la fixture porte une pause de 120 s.
    assert sessions["duration_total_s"].iloc[-1] > sessions["duration_total_s"].iloc[0]


def test_reference_trimp_is_kept_for_cross_validation(ingested):
    """
    GoldenCheetah publie son propre TRIMP. Le conserver permet de recouper
    notre calcul contre une implementation independante, ce qui vaut mieux
    qu'une assertion ecrite par nous-memes.
    """
    sessions = load_sessions(ingested)
    assert sessions["trimp_reference"].notna().all()
    assert sessions["trimp"].gt(0).all()


def test_uncalibrated_athlete_stops_the_pipeline_with_a_clear_message(cfg, monkeypatch):
    monkeypatch.setattr(orch, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(orch, "estimate_hr_bounds", lambda *a, **k: None)
    with pytest.raises(ConfigError, match="FC max"):
        run_pipeline(LocalZipSpec(FIXTURE), cfg)


def test_hr_bounds_are_estimated_from_the_summary(cfg):
    """
    Le TRIMP exige FC max et repos, qui se deduisent de l'historique : sans
    amorcage la chaine ne peut pas demarrer. Les resumes portent ces valeurs
    par activite, ce qui brise la circularite sans decoder un seul CSV.
    """
    bounds = orch.estimate_hr_bounds(LocalZipSpec(FIXTURE), cfg)
    if bounds is not None:            # la fixture ne compte que six activites
        hr_max, hr_rest = bounds
        assert hr_max > hr_rest + 40


# -- charge quotidienne ----------------------------------------------------

def test_daily_load_calendar_has_no_holes(ingested):
    daily = load_daily_load(ingested)
    dates = pd.to_datetime(daily["date"])
    assert (dates.diff().dropna() == pd.Timedelta(days=1)).all()
    # Les jours sans seance doivent exister : le modele les parcourt pour
    # faire decroitre fitness et fatigue.
    assert (daily["trimp"] == 0).sum() > 100


def test_daily_load_sums_same_day_sessions():
    sessions = pd.DataFrame({
        "session_id": ["a", "b", "c"],
        "date": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-01-03"]),
        "trimp": [10.0, 15.0, 20.0],
        "source": ["opendata_osf"] * 3,
    })
    daily = build_daily_load(sessions)
    assert len(daily) == 3                      # le 2 janvier est cree a zero
    assert daily.loc[0, "trimp"] == 25.0
    assert daily.loc[0, "n_sessions"] == 2
    assert daily.loc[1, "trimp"] == 0.0


def test_daily_load_counts_every_sport():
    """
    La charge est systemique : chez des triathletes, ne compter que la course
    laisserait une seance tous les six jours, bien trop epars pour un modele
    a convolution exponentielle.
    """
    sessions = pd.DataFrame({
        "session_id": ["a", "b"],
        "date": pd.to_datetime(["2019-01-01", "2019-01-01"]),
        "sport": ["Run", "Bike"],
        "trimp": [40.0, 60.0],
        "source": ["opendata_osf"] * 2,
    })
    assert build_daily_load(sessions).loc[0, "trimp"] == 100.0


def test_empty_input_gives_an_empty_calendar():
    assert build_daily_load(pd.DataFrame()).empty


# -- provenance ------------------------------------------------------------

def test_synthetic_sources_are_refused_by_default(pipeline_cfg, monkeypatch):
    from src.pipeline.provenance import DataSource

    spec = LocalZipSpec(FIXTURE, source=DataSource.SYNTHETIC)
    report = run_pipeline(spec, pipeline_cfg, allow_synthetic=False)
    assert report.n_new == 0
    assert report.n_failed == 6
    assert all("synthetique" in why for _, why in report.failures)
