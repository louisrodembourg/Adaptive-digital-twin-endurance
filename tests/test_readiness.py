"""
Maturite : ce que le systeme peut affirmer, et quand.

Le verdict de dose-reponse demande 180 jours de contexte puis huit plis de
28 jours. Avant cela `assess_dose_response` rend « indeterminable », ce qui
est exact et inutile : un nouvel utilisateur n'entend rien pendant plus d'un
an sans savoir ni pourquoi, ni combien de temps encore.

Ces tests verrouillent deux choses. D'abord que chaque barreau de l'echelle
porte le seuil que le code impose VRAIMENT — un seuil recopie ici derive le
jour ou la fonction change, et l'echelle devient un mensonge rassurant.
Ensuite que le prior de population se lit dans l'artefact de cohorte : un
chiffre de population en dur devient faux des que la cohorte grandit, et
personne ne s'en apercoit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.readiness import (
    CHRONIC_DAYS,
    FITNESS_SETTLING_DAYS,
    FORM_DAYS,
    Capability,
    assess_readiness,
    format_readiness,
)
from src.evaluation.validation import horizon_splits


def _daily(n, start="2025-01-01", trimp=50.0):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D"),
        "trimp": trimp,
    })


def _observations(n, start="2025-01-01", step_days=3, value=250.0):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq=f"{step_days}D"),
        "pi": value + rng.normal(0, 5.0, n),
    })


def _cohort(detected=4, total=13):
    """Un artefact de cohorte reduit a ce que le prior lit."""
    lows = [0.1] * detected + [-0.3] * (total - detected)
    return pd.DataFrame({
        "uuid": [f"a{i}" for i in range(total)],
        "variante": "velo seul",
        "reference": "moyenne",
        "skill": [0.3] * detected + [-0.05] * (total - detected),
        "ci_low": lows,
        "ci_high": [0.6] * total,
    })


# ── le vide se decrit ────────────────────────────────────────────────────

def test_a_system_with_no_data_describes_its_own_emptiness():
    """Un systeme qui demarre doit pouvoir dire qu'il n'a rien, sans lever."""
    readiness = assess_readiness(pd.DataFrame(), pd.DataFrame())
    assert readiness.days == 0
    assert readiness.observations == 0
    assert readiness.folds == 0
    assert readiness.available == ()


def test_nothing_is_claimed_available_on_an_empty_system():
    text = format_readiness(assess_readiness(pd.DataFrame(), pd.DataFrame()))
    assert "0/6" in text
    assert "[oui]" not in text


# ── chaque barreau porte le seuil que le code impose ─────────────────────

def test_the_load_curve_exists_from_the_first_day():
    readiness = assess_readiness(_daily(1), pd.DataFrame())
    assert readiness.capability("charge (fitness, fatigue)").available


def test_an_unsettled_fitness_curve_says_so():
    """
    Avant une constante de temps, la moyenne mobile exponentielle depend
    encore surtout de sa valeur initiale. L'afficher sans le dire laisserait
    croire a une forme mesuree.
    """
    young = assess_readiness(_daily(FITNESS_SETTLING_DAYS - 1), pd.DataFrame())
    settled = assess_readiness(_daily(FITNESS_SETTLING_DAYS + 5), pd.DataFrame())
    assert "pas encore stabilisee" in young.capability("charge (fitness, fatigue)").detail
    assert "stabilisee depuis" in settled.capability("charge (fitness, fatigue)").detail


@pytest.mark.parametrize(("name", "threshold"), [
    ("borne aigu/chronique", CHRONIC_DAYS),
    ("seuils de fraicheur propres", FORM_DAYS),
])
def test_a_calendar_threshold_unlocks_exactly_at_its_value(name, threshold):
    """
    Les seuils viennent de `_chronic_daily_load` et de `form_thresholds`. Si
    l'un change la, ce test echoue ici — c'est le but.
    """
    assert not assess_readiness(_daily(threshold - 1), pd.DataFrame()).capability(name).available
    assert assess_readiness(_daily(threshold), pd.DataFrame()).capability(name).available


def test_a_missing_calendar_threshold_counts_the_days_left():
    readiness = assess_readiness(_daily(10), pd.DataFrame())
    assert readiness.capability("borne aigu/chronique").missing == (
        f"{CHRONIC_DAYS - 10} jour(s) de calendrier")


def test_the_envelope_needs_its_minimum_of_measures():
    """`rolling_envelope` ecarte une fenetre qui compte moins de trois mesures."""
    daily = _daily(200)
    two = assess_readiness(daily, _observations(2), min_periods=3)
    three = assess_readiness(daily, _observations(3), min_periods=3)
    assert not two.capability("enveloppe de performance").available
    assert "1 mesure(s)" in two.capability("enveloppe de performance").missing
    assert three.capability("enveloppe de performance").available


def test_an_observation_column_that_is_absent_is_not_a_crash():
    readiness = assess_readiness(_daily(100), pd.DataFrame({"date": [], "autre": []}))
    assert readiness.observations == 0
    assert not readiness.capability("enveloppe de performance").available


# ── les plis ne sont pas recomptes a la main ─────────────────────────────

def test_the_fold_count_is_the_evaluator_s_own():
    """
    Le compte passe par `horizon_splits`, la fonction que l'evaluation
    utilise. Le reimplementer ici laisserait l'echelle et le protocole
    diverger sans que rien ne le signale.
    """
    daily, observations = _daily(900), _observations(300, step_days=3)
    readiness = assess_readiness(daily, observations)

    from src.layer_a_signal.envelope import EnvelopeSpec, rolling_envelope
    envelope = rolling_envelope(
        observations.rename(columns={"pi": "pi"}),
        EnvelopeSpec("pi", 42, min_periods=3), calendar="daily")
    expected = len(list(horizon_splits(envelope["date"], horizon_days=28,
                                       min_train_days=180, step_days=28)))
    assert readiness.folds == expected


def test_a_forecast_becomes_evaluable_at_the_first_fold():
    daily = _daily(900)
    short = assess_readiness(daily, _observations(30, step_days=3))
    long = assess_readiness(daily, _observations(300, step_days=3))
    assert not short.capability("prevision evaluable").available
    assert long.capability("prevision evaluable").available


def test_the_verdict_counts_the_days_it_still_needs():
    """
    C'est la partie actionnable : « indeterminable » ne dit rien, « il manque
    224 jours » dit quoi attendre.
    """
    readiness = assess_readiness(_daily(900), pd.DataFrame(), min_folds=8,
                                 horizon_days=28)
    assert readiness.capability("verdict de dose-reponse").missing == "224 jour(s) (8 pli(s))"


def test_a_mature_athlete_has_every_capability():
    readiness = assess_readiness(_daily(900), _observations(300, step_days=3))
    assert len(readiness.available) == len(readiness.capabilities)
    assert "6/6" in format_readiness(readiness)


# ── le prior de population ───────────────────────────────────────────────

def test_the_prior_is_read_from_the_cohort_and_not_recopied():
    """
    Un chiffre de population en dur devient faux des que la cohorte grandit.
    Changer l'artefact doit changer le prior.
    """
    four = assess_readiness(_daily(10), pd.DataFrame(), cohort=_cohort(4, 13))
    seven = assess_readiness(_daily(10), pd.DataFrame(), cohort=_cohort(7, 20))
    assert "4 athlete(s) sur 13" in four.prior
    assert "7 athlete(s) sur 20" in seven.prior
    assert "chez 9, non" in four.prior


def test_no_cohort_means_no_prior_rather_than_an_invented_one():
    assert assess_readiness(_daily(10), pd.DataFrame()).prior is None
    assert assess_readiness(_daily(10), pd.DataFrame(),
                            cohort=pd.DataFrame()).prior is None


def test_a_missing_cohort_file_is_not_an_error(tmp_path):
    readiness = assess_readiness(_daily(10), pd.DataFrame(),
                                 cohort=tmp_path / "absent.parquet")
    assert readiness.prior is None


def test_a_cohort_file_is_read_from_disk(tmp_path):
    path = tmp_path / "cohort.parquet"
    _cohort(4, 13).to_parquet(path, index=False)
    readiness = assess_readiness(_daily(10), pd.DataFrame(), cohort=path)
    assert readiness.prior is not None
    assert "4 athlete(s) sur 13" in readiness.prior


def test_the_prior_is_offered_only_while_the_verdict_is_missing():
    """
    Un athlete qui a son propre verdict n'a pas besoin du prior ; le lui
    servir quand meme brouillerait ce qui le concerne.
    """
    young = format_readiness(assess_readiness(_daily(10), pd.DataFrame(),
                                              cohort=_cohort()))
    mature = format_readiness(assess_readiness(_daily(900),
                                               _observations(300, step_days=3),
                                               cohort=_cohort()))
    assert "En attendant son propre verdict" in young
    assert "En attendant son propre verdict" not in mature


# ── rendu ────────────────────────────────────────────────────────────────

def test_an_available_capability_shows_no_missing_line():
    available = Capability("x", available=True, detail="fait", missing="")
    assert "manque" not in available.describe()


def test_an_unavailable_capability_shows_what_is_missing():
    blocked = Capability("x", available=False, detail="pas fait", missing="12 jours")
    assert "manque : 12 jours" in blocked.describe()
    assert "[non]" in blocked.describe()
