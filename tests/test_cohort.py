"""
Replication sur une cohorte.

Un resultat sur un athlete est une anecdote. Ces tests verrouillent la
mecanique qui le confronte aux autres athletes du cache, et surtout la
distinction que la mesure a rendue necessaire : une minorite d'athletes montre
un effet et le reste rien, ce qui peut signifier « effet universel mal mesure »
ou « effet reel chez certains seulement ». Les deux ne menent pas au meme
modele, et `detection_contrast` est ce qui tranche.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.evaluation.cohort import (
    POWER_KEY,
    TRIMP_KEYS,
    CohortCriteria,
    _variant,
    detection_contrast,
    evaluate_athlete,
    format_cohort,
    format_contrast,
    screen_cache,
    summarise_cohort,
    summary_to_frames,
)
from src.ingestion.opendata import parse_summary


def _summary(uuid="abc", n=400, start="2018-01-01", sport="Bike",
             trimp_key="trimp_points", power=250.0, step_days=2):
    """Un resume JSON comme en produit GoldenCheetah, reduit a l'essentiel."""
    first = datetime.fromisoformat(start)
    rides = []
    for i in range(n):
        when = first + timedelta(days=i * step_days)
        rides.append({
            "filename": f"{i}.csv",
            "date": when.strftime("%Y/%m/%d %H:%M:%S UTC"),
            "sport": sport,
            "data": "hp",
            "METRICS": {trimp_key: 80.0 + (i % 17),
                        POWER_KEY: power + (i % 23)},
        })
    return parse_summary({"ATHLETE": {"gender": "M", "yob": "1985"},
                          "RIDES": rides}, uuid)


def _zip(tmp_path, summary_raw, name="athlete.zip"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("abc.json", json.dumps(summary_raw))
    return path


def _raw(n=400, sport="Bike", trimp_key="trimp_points", power=250.0, step_days=2):
    first = datetime.fromisoformat("2018-01-01")
    rides = []
    for i in range(n):
        when = first + timedelta(days=i * step_days)
        rides.append({
            "filename": f"{i}.csv",
            "date": when.strftime("%Y/%m/%d %H:%M:%S UTC"),
            "sport": sport,
            "data": "hp",
            "METRICS": {trimp_key: 80.0 + (i % 17),
                        POWER_KEY: power + (i % 23)},
        })
    return {"ATHLETE": {"gender": "M", "yob": "1985"}, "RIDES": rides}


# -- lecture des resumes ---------------------------------------------------

def test_a_summary_yields_sessions_and_power_without_opening_any_csv():
    """
    Tout ce dont l'experience a besoin est dans le JSON. Decoder les CSV de
    62 athletes couterait des heures pour la meme reponse.
    """
    frames = summary_to_frames(_summary(n=100))
    assert frames is not None
    assert len(frames.sessions) == 100
    assert len(frames.power) == 100
    assert set(frames.sessions.columns) >= {"session_id", "date", "sport",
                                            "trimp", "source"}
    assert frames.trimp_key == "trimp_points"


def test_the_load_metric_falls_back_before_giving_up():
    """`trimp_zonal_points` couvre quelques seances de plus et sert de repli."""
    frames = summary_to_frames(_summary(n=60, trimp_key="trimp_zonal_points"))
    assert frames is not None
    assert frames.trimp_key == "trimp_zonal_points"
    assert frames.trimp_key in TRIMP_KEYS


def test_an_athlete_without_any_load_metric_is_refused():
    """Sans charge il n'y a pas d'experience a mener, et le dire vaut mieux."""
    assert summary_to_frames(_summary(n=60, trimp_key="bidon_inexistant")) is None


def test_a_zero_power_is_a_missing_sensor_not_a_weak_athlete():
    """GoldenCheetah ecrit 0 quand aucun capteur n'a enregistre."""
    frames = summary_to_frames(_summary(n=50, power=0.0))
    assert frames is not None
    assert frames.power.empty


def test_an_absurd_power_is_dropped_too():
    frames = summary_to_frames(_summary(n=50, power=5000.0))
    assert frames is not None and frames.power.empty


def test_running_sessions_carry_load_but_no_power_target():
    """
    Une cible velo ne se nourrit pas d'une seance de course : la charge
    compte, la puissance non.
    """
    frames = summary_to_frames(_summary(n=80, sport="Run"))
    assert frames is not None
    assert len(frames.sessions) == 80
    assert frames.power.empty


def test_virtualride_feeds_the_power_target():
    """Un watt est un watt : sur home-trainer comme sur route."""
    frames = summary_to_frames(_summary(n=80, sport="VirtualRide"))
    assert frames is not None and len(frames.power) == 80


# -- criblage --------------------------------------------------------------

def test_the_cache_is_screened_on_declared_criteria(tmp_path):
    _zip(tmp_path, _raw(n=400), "bon.zip")
    kept = screen_cache(tmp_path, CohortCriteria())
    assert len(kept) == 1
    assert kept[0].span_days >= 540


@pytest.mark.parametrize(("criteria", "raw"), [
    (CohortCriteria(min_power_observations=500), _raw(n=400)),
    (CohortCriteria(min_span_days=10_000), _raw(n=400)),
    (CohortCriteria(min_rides_with_load=5_000), _raw(n=400)),
])
def test_an_athlete_below_a_criterion_is_left_out(tmp_path, criteria, raw):
    _zip(tmp_path, raw)
    assert screen_cache(tmp_path, criteria) == []


def test_a_short_history_cannot_carry_a_horizon_protocol(tmp_path):
    """
    Le seuil de duree n'est pas un gout : il faut `min_train_days` de contexte
    plus plusieurs plis pour que la validation dise quelque chose.
    """
    _zip(tmp_path, _raw(n=40, step_days=1))
    assert screen_cache(tmp_path, CohortCriteria()) == []


def test_an_unreadable_archive_does_not_take_the_cohort_down(tmp_path):
    (tmp_path / "casse.zip").write_bytes(b"ceci n'est pas un zip")
    _zip(tmp_path, _raw(n=400), "bon.zip")
    assert len(screen_cache(tmp_path, CohortCriteria())) == 1


# -- noms de variantes ----------------------------------------------------

def test_short_names_resolve_to_registered_variants():
    """
    Si ce mappage derivait, les paires disparaitraient silencieusement et la
    replication ne rapporterait rien — la panne ressemblerait a un resultat.
    """
    from src.evaluation.report import build_variants

    daily = pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=400, freq="D"),
        "trimp_run": 30.0, "trimp_bike": 60.0, "trimp_other": 0.0,
    }).assign(trimp=90.0)
    registered = set(build_variants(daily))
    for short in ("velo seul", "course seule", "moyenne", "persistance"):
        assert _variant(short, 42.0, 12.0) in registered


# -- agregation -----------------------------------------------------------

def _table(skills, lows, highs, reference="moyenne"):
    return pd.DataFrame({
        "uuid": [f"a{i}" for i in range(len(skills))],
        "variante": "velo seul",
        "reference": reference,
        "r2_variante": skills,
        "r2_reference": 0.0,
        "n_jours": 800,
        "n_puissances": 300,
        "plis": 20,
        "skill": skills,
        "ci_low": lows,
        "ci_high": highs,
        "p_a_better": 0.9,
        "n_blocks": 20,
    })


def test_the_summary_reports_a_distribution_not_a_mean():
    """
    Une moyenne de competences sur des historiques tres inegaux cacherait
    justement ce qu'on cherche : combien d'athletes voient l'effet.
    """
    table = _table([0.4, 0.3, -0.05, -0.1, 0.02],
                   [0.1, 0.05, -0.5, -0.6, -0.3],
                   [0.7, 0.6, 0.3, 0.2, 0.4])
    summary = summarise_cohort(table)
    row = summary.iloc[0]
    assert row["athletes"] == 5
    assert row["gain_positif"] == 3
    assert row["ic_exclut_zero_pour"] == 2
    assert row["ic_exclut_zero_contre"] == 0
    assert "gain_moyen" not in summary.columns


def test_a_unanimous_cohort_is_distinguished_from_a_split_one():
    unanimous = summarise_cohort(_table([0.3] * 10, [0.1] * 10, [0.5] * 10))
    split = summarise_cohort(_table([0.3, -0.3] * 5, [-0.2] * 10, [0.5] * 10))
    assert unanimous.iloc[0]["p_signe"] < 0.01
    assert split.iloc[0]["p_signe"] == pytest.approx(1.0)


def test_an_empty_cohort_claims_nothing():
    assert summarise_cohort(pd.DataFrame()).empty
    assert "Aucun athlete" in format_cohort(pd.DataFrame(), pd.DataFrame())


# -- le contraste qui tranche ---------------------------------------------

def test_an_undetected_group_centred_on_zero_means_real_heterogeneity():
    """
    Le test qui distingue les deux lectures. Si l'effet etait universel mais
    noye, les estimations ponctuelles des non detectes resteraient positives.
    Centrees sur zero, elles disent que l'effet n'existe que chez certains.
    """
    table = _table([0.4, 0.35, -0.05, 0.02, -0.08],
                   [0.1, 0.1, -0.6, -0.5, -0.7],
                   [0.7, 0.6, 0.5, 0.6, 0.5])
    contrast = detection_contrast(table)
    detected = contrast[contrast["groupe"] == "detectes"].iloc[0]
    other = contrast[contrast["groupe"] == "non detectes"].iloc[0]
    assert detected["athletes"] == 2
    assert detected["gain_median"] > 0.3
    assert abs(other["gain_median"]) < 0.1
    assert "heterogeneite" in format_contrast(contrast).lower()


def test_a_positive_but_drowned_group_is_not_called_heterogeneous():
    """
    Le cas oppose doit se lire differemment : des non detectes nettement
    positifs signalent un defaut de puissance, pas une heterogeneite.
    """
    table = _table([0.5, 0.45, 0.3, 0.35, 0.28],
                   [0.2, 0.2, -0.1, -0.05, -0.2],
                   [0.8, 0.7, 0.7, 0.8, 0.7])
    contrast = detection_contrast(table)
    other = contrast[contrast["groupe"] == "non detectes"].iloc[0]
    assert other["gain_median"] > 0.25
    assert other["gain_positif"] == other["athletes"]


def test_the_contrast_needs_the_right_reference():
    assert detection_contrast(_table([0.1], [0.0], [0.2]), "persistance").empty
    assert detection_contrast(pd.DataFrame()).empty


# -- protocole -------------------------------------------------------------

def test_an_athlete_with_too_few_folds_publishes_nothing():
    """
    Une variante qui n'a presque rien execute ne publie rien, et la regle
    vaut pour un athlete entier.
    """
    frames = summary_to_frames(_summary(n=400))
    assert frames is not None
    table = evaluate_athlete(frames, criteria=CohortCriteria(min_folds=10_000))
    assert table.empty
