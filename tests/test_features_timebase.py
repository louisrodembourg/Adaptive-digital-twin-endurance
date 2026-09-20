"""
Layer A : les metriques doivent raisonner en secondes reelles.

Tout le module supposait implicitement une ligne par seconde sans trou. Les
CSV GoldenCheetah portent une colonne `secs` explicite qui rend cette
hypothese fausse des qu'il y a une pause.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.layer_a_signal import features
from src.layer_a_signal.timebase import (
    active_duration_s,
    attach_timebase,
    longest_contiguous_s,
)


@pytest.fixture
def calibrated(monkeypatch):
    base = load_config()
    cfg = replace(
        base,
        athlete=replace(base.athlete, hr_max=190.0, hr_rest=50.0),
        goal=replace(base.goal, reference_time_s=4920.0),
    )
    monkeypatch.setattr(features, "load_config", lambda *a, **k: cfg)
    return cfg


def _session(n=1200, gap_after=None, gap_s=600, speed=4.5, hr_from=130, hr_to=175):
    secs = list(range(n))
    if gap_after is not None:
        secs = list(range(gap_after)) + list(range(gap_after + gap_s, gap_after + gap_s + n - gap_after))
    return pd.DataFrame({
        "secs": secs,
        "hr": np.linspace(hr_from, hr_to, n),
        "speed_ms": np.full(n, speed),
        "speed_kmh": np.full(n, speed * 3.6),
        "altitude": np.zeros(n),
    })


# -- base de temps ---------------------------------------------------------

def test_gap_is_capped_not_counted_in_full():
    df = attach_timebase(_session(1200, gap_after=600, gap_s=600), max_gap_s=30)
    # 1199 pas de 1 s + un trou ramene a 30 s.
    assert active_duration_s(df) == pytest.approx(1199 + 30)


def test_without_a_time_column_one_hertz_is_assumed_explicitly():
    assert active_duration_s(pd.DataFrame({"hr": [150.0] * 42})) == 42.0


def test_duplicate_timestamps_carry_no_load():
    df = attach_timebase(pd.DataFrame({"secs": [0, 0, 1], "hr": [150.0] * 3}))
    assert df["dt_s"].tolist() == [1.0, 0.0, 1.0]


def test_longest_contiguous_ignores_scattered_samples():
    dt = pd.Series([1.0] * 200)
    scattered = pd.Series([True, False] * 100)
    sustained = pd.Series([True] * 90 + [False] * 110)
    assert longest_contiguous_s(scattered, dt) == 1.0
    assert longest_contiguous_s(sustained, dt) == 90.0


# -- duree de seance -------------------------------------------------------

def test_session_duration_is_not_the_row_count(calibrated):
    f = features.session_features(_session(1200, gap_after=600, gap_s=600))
    assert f["duration_total_s"] == 1229  # et non 1200
    assert not isinstance(f.get("hr_zones"), dict)  # zones aplaties pour Parquet


def test_hr_zone_seconds_sum_to_the_active_duration(calibrated):
    f = features.session_features(_session(600))
    total = sum(v for k, v in f.items() if k.startswith("hr_zone") and k.endswith("_s"))
    assert total == pytest.approx(f["duration_total_s"], abs=2)


# -- GAP -------------------------------------------------------------------

@pytest.mark.parametrize("grade_pct,expected", [(5.0, 0.165), (-5.0, -0.090), (0.0, 0.0)])
def test_gap_corrects_by_percent_not_by_fraction(grade_pct, expected):
    """
    Les coefficients de Daniels s'appliquent par point de pourcentage. Les
    appliquer a une pente exprimee en fraction sous-corrigeait d'un facteur
    cent, ce qui rendait le GAP indiscernable de la vitesse brute.
    """
    n, speed = 200, 3.0
    alt = np.arange(n) * speed * grade_pct / 100.0
    df = pd.DataFrame({"secs": range(n), "speed_ms": [speed] * n, "altitude": alt})
    factor = features.grade_adjusted_pace(df)["speed_gap_ms"].iloc[5:].mean() / speed
    assert factor - 1.0 == pytest.approx(expected, abs=0.005)


def test_gap_uses_real_elapsed_time_for_distance():
    # Meme denivele, meme vitesse, mais un pas de temps de 5 s : la pente
    # reelle est cinq fois plus faible que ce qu'un calcul a 1 Hz donnerait.
    n, speed = 100, 3.0
    alt = np.arange(n) * 0.75  # 0.75 m tous les 5 s a 3 m/s -> 5 %
    df = pd.DataFrame({"secs": np.arange(n) * 5, "speed_ms": [speed] * n, "altitude": alt})
    factor = features.grade_adjusted_pace(df)["speed_gap_ms"].iloc[5:].mean() / speed
    assert factor - 1.0 == pytest.approx(0.165, abs=0.01)


# -- EF specifique ---------------------------------------------------------

def test_specific_ef_requires_a_sustained_effort(calibrated):
    target = calibrated.goal.target_speed_ms
    n = 600
    # 120 secondes a l'allure cible, mais une seconde sur deux.
    speed = np.where(np.arange(n) % 2 == 0, target, target * 0.5)
    scattered = pd.DataFrame({"secs": range(n), "speed_ms": speed, "hr": [160.0] * n})
    assert np.isnan(features.specific_ef(scattered))

    sustained = pd.DataFrame({
        "secs": range(n),
        "speed_ms": np.where(np.arange(n) < 120, target, target * 0.5),
        "hr": [160.0] * n,
    })
    assert not np.isnan(features.specific_ef(sustained))


# -- DFA -------------------------------------------------------------------

def test_dfa_returns_nan_without_real_rr_intervals(calibrated):
    """
    La fonction fabriquait auparavant des RR par un AR(1) a graine fixe et
    renvoyait le resultat comme une mesure ordinaire, qui atteignait la
    soupape de securite et pouvait bloquer une seance intense.
    """
    assert np.isnan(features.compute_dfa_alpha1(_session(600)))


def test_dfa_works_on_real_rr_intervals(calibrated):
    from tests.fixtures.synthetic import synthetic_rr_from_hr

    n = 600
    df = _session(n)
    df["rr_ms"] = synthetic_rr_from_hr(df["hr"].to_numpy())
    assert not np.isnan(features.compute_dfa_alpha1(df))


def test_production_code_cannot_fabricate_rr():
    from src.layer_a_signal import features_advanced

    assert not hasattr(features_advanced, "generate_mock_rr")


def test_invented_strength_trimp_is_gone():
    assert not hasattr(features, "compute_trimp_strength")
