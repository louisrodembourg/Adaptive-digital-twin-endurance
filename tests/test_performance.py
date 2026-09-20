"""
Index de performance : la piece qui manquait pour alimenter Banister.

Le modele apprend en confrontant une charge accumulee a des performances
mesurees. Le projet n'en produisait aucune, d'ou un chemin PyMC reste mort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.layer_a_signal.performance import (
    DEFAULT_DURATIONS_S,
    PerformanceError,
    ReferenceCurve,
    fit_reference_curve,
    mean_max_speed,
    performance_index,
    plausible_speed_ceiling,
    target_index,
)


def _run(duration_s=1800, speed_ms=3.5, surge=None, gap=None):
    """Sortie a vitesse constante, avec accelaration ou pause optionnelle."""
    n = duration_s
    v = np.full(n, speed_ms)
    if surge:
        start, length, value = surge
        v[start:start + length] = value
    secs = np.arange(n, dtype=float)
    if gap:
        at, length = gap
        secs[at:] += length
    km = np.cumsum(v) / 1000.0
    return pd.DataFrame({"secs": secs, "km": km, "speed_ms": v, "hr": np.full(n, 160.0)})


# -- vitesse maximale moyenne ---------------------------------------------

def test_constant_effort_gives_that_speed_at_every_duration():
    mmp = mean_max_speed(_run(1800, 3.5), (300, 600, 1200))
    for d, v in mmp.items():
        assert v == pytest.approx(3.5, abs=0.02), d


def test_a_surge_is_picked_up_at_the_matching_duration():
    mmp = mean_max_speed(_run(1800, 3.0, surge=(600, 300, 4.5)), (300, 1200))
    assert mmp[300] == pytest.approx(4.5, abs=0.05)
    assert mmp[1200] < 4.5  # dilue sur une fenetre plus longue


def test_durations_longer_than_the_session_are_nan():
    assert np.isnan(mean_max_speed(_run(400), (300, 1800))[1800])


def test_a_pause_cannot_manufacture_a_record():
    """
    Pendant une pause la distance n'avance pas mais le temps si. Une fenetre
    qui enjambe le trou doit etre ecartee, sans quoi elle compare la distance
    d'un effort a la duree d'un arret.
    """
    paused = _run(1200, 3.0, gap=(600, 900))
    assert mean_max_speed(paused, (300,))[300] == pytest.approx(3.0, abs=0.1)


def test_implausible_speeds_are_rejected_not_recorded():
    # Vu sur donnees reelles : des seances etiquetees Run sont en fait des
    # enchainements velo-course, dont le canal distance donne 40-58 km/h.
    absurd = _run(1200, 16.0)
    assert np.isnan(mean_max_speed(absurd, (300,))[300])


def test_the_ceiling_tightens_with_duration():
    assert plausible_speed_ceiling(300) > plausible_speed_ceiling(3600)
    assert plausible_speed_ceiling(3600) > 5.0  # sans exclure un athlete reel


def test_speed_is_derived_from_speed_column_when_distance_is_absent():
    df = _run(900, 3.2).drop(columns=["km"])
    assert mean_max_speed(df, (300,))[300] == pytest.approx(3.2, abs=0.05)


# -- courbe de reference ---------------------------------------------------

def _history(n=40, base=4.0, exponent=0.07):
    """Historique synthetique suivant exactement une loi de Riegel."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        scale = rng.uniform(0.80, 1.0)  # seances plus ou moins engagees
        out.append({d: base * (d / 300.0) ** (-exponent) * scale
                    for d in DEFAULT_DURATIONS_S})
    return out


def test_fit_recovers_the_generating_exponent():
    curve = fit_reference_curve(_history())
    assert curve.exponent == pytest.approx(0.07, abs=0.01)
    assert curve.r_squared > 0.99


def test_fit_is_robust_to_a_single_corrupt_session():
    """
    Le maximum absolu rendait la courbe otage d'une seule seance : sur
    l'athlete reel, quatre fichiers sur soixante suffisaient a produire une
    reference a 15 m/s. Le percentile absorbe l'aberration.
    """
    history = _history()
    history.append(dict.fromkeys(DEFAULT_DURATIONS_S, 12.0))
    curve = fit_reference_curve(history)
    assert curve.exponent == pytest.approx(0.07, abs=0.015)
    assert curve.speed_at(300) < 6.0


def test_sparse_durations_are_excluded():
    # Une duree soutenue par une poignee de seances ferait du percentile un
    # maximum deguise, et l'exposant ajuste pouvait ressortir negatif.
    history = [{300: 4.0, 600: 3.8, 1200: 3.6, 1800: 3.5} for _ in range(30)]
    history[0][3600] = 9.0
    curve = fit_reference_curve(history)
    assert curve.speed_at(3600) < 4.0


def test_too_few_durations_is_an_explicit_error():
    with pytest.raises(PerformanceError, match="duree"):
        fit_reference_curve([{300: 4.0}] * 30)


def test_distance_predictions_are_ordered_and_plausible():
    curve = fit_reference_curve(_history())
    t5, t21, t42 = (curve.time_for_distance(d) for d in (5000, 21097.5, 42195))
    assert t5 < t21 < t42
    assert 2.0 < t42 / t21 < 2.4  # un marathon vaut un peu plus de deux semis


def test_curve_survives_a_round_trip():
    curve = fit_reference_curve(_history())
    assert ReferenceCurve.from_dict(curve.to_dict()).speed_at(600) == pytest.approx(
        curve.speed_at(600)
    )


# -- observation -----------------------------------------------------------

@pytest.fixture
def curve():
    return fit_reference_curve(_history())


def _observe(df, curve, **kw):
    return performance_index(df, curve, session_id="s", date=pd.Timestamp("2019-01-01"),
                             hr_max=190.0, **kw)


def test_a_maximal_effort_becomes_an_observation(curve):
    df = _run(1800, curve.speed_at(1800) * 1.02)
    df["hr"] = 185.0
    obs = _observe(df, curve)
    assert obs is not None and obs.index > 1.0 and obs.is_clean


def test_an_easy_run_is_not_a_performance(curve):
    """
    Alimenter la vraisemblance avec un footing etiquete performance est le
    moyen le plus sur d'obtenir un posterior denue de sens.
    """
    df = _run(1800, curve.speed_at(1800) * 0.7)
    assert _observe(df, curve) is None


def test_fast_without_cardiac_cost_is_flagged(curve):
    df = _run(1800, curve.speed_at(1800) * 1.05)
    df["hr"] = 120.0          # descente, vent favorable, ou capteur douteux
    obs = _observe(df, curve)
    assert obs is not None and "hr_not_maximal" in obs.quality_flags


def test_hilly_sessions_are_flagged(curve):
    df = _run(1800, curve.speed_at(1800) * 1.02)
    df["hr"] = 185.0
    obs = _observe(df, curve, elevation_gain_m=600.0, distance_km=10.0)
    assert obs is not None and "hilly" in obs.quality_flags


def test_missing_heart_rate_is_flagged(curve):
    df = _run(1800, curve.speed_at(1800) * 1.02).drop(columns=["hr"])
    obs = _observe(df, curve)
    assert obs is not None and "no_hr" in obs.quality_flags


# -- objectif --------------------------------------------------------------

def test_goal_translates_into_an_index_without_hardcoded_constants(curve):
    """
    Remplace les deux constantes 4920 et 4680 : la cible se derive de la
    courbe de l'athlete, donc n'importe quel athlete et n'importe quel
    objectif fonctionnent.
    """
    current = curve.time_for_distance(21097.5)
    assert target_index(curve, 21097.5, current) == pytest.approx(1.0, abs=0.01)
    assert target_index(curve, 21097.5, current * 0.95) == pytest.approx(1.05, abs=0.02)
    assert target_index(curve, 21097.5, current * 1.05) < 1.0


def test_nonpositive_target_is_rejected(curve):
    with pytest.raises(PerformanceError):
        target_index(curve, 21097.5, 0)
