"""
Protocole d'evaluation.

Ce module existe parce que les scores d'ajustement m'ont induit en erreur :
ils suggeraient un modele « faible mais positif » (R2 de 0,03 a 0,14 selon
les variantes) alors que hors echantillon aucune ne bat la moyenne. Ses
tests verifient donc surtout qu'il ne peut PAS flatter un modele.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines import (
    EwmaBaseline,
    LastValueBaseline,
    MeanBaseline,
    TrendBaseline,
    default_baselines,
)
from src.evaluation.metrics import (
    autocorrelation,
    interval_coverage,
    mae,
    r2,
    rmse,
    skill_score,
)
from src.evaluation.validation import (
    blocked_splits,
    compare,
    cross_validate,
    expanding_window_splits,
)


@pytest.fixture
def dates():
    return pd.Series(pd.date_range("2019-01-01", periods=60, freq="7D"))


# -- metriques -------------------------------------------------------------

def test_perfect_prediction_scores_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == pytest.approx(1.0)
    assert rmse(y, y) == 0.0


def test_predicting_the_mean_scores_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, np.full(4, y.mean())) == pytest.approx(0.0)


def test_worse_than_the_mean_is_negative():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, np.array([4.0, 3.0, 2.0, 1.0])) < 0


def test_r2_uses_the_reference_it_is_given():
    """
    Hors echantillon, la reference doit etre la moyenne connue au moment de
    predire, pas celle du jeu de test : cette derniere donnerait au modele
    une information dont il ne disposait pas.
    """
    y = np.array([5.0, 6.0, 7.0])
    pred = np.full(3, 6.0)
    # Sans reference : denominateur = variance du test, prediction = sa moyenne.
    assert r2(y, pred) == pytest.approx(0.0)
    # Avec une reference decalee, le denominateur change et le score aussi.
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - 4.0) ** 2).sum())
    assert r2(y, pred, baseline_mean=4.0) == pytest.approx(1 - ss_res / ss_tot)


def test_metrics_ignore_missing_values():
    y = np.array([1.0, np.nan, 3.0])
    assert np.isfinite(rmse(y, np.array([1.0, 99.0, 3.0])))
    assert mae(y, np.array([1.0, 99.0, 3.0])) == pytest.approx(0.0)


def test_skill_score_reads_as_a_gain_over_a_reference():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    better = np.array([1.1, 2.1, 2.9, 3.9])
    reference = np.full(4, y.mean())
    assert skill_score(y, better, reference) > 0
    assert skill_score(y, reference, reference) == pytest.approx(0.0)


def test_autocorrelation_separates_noise_from_structure():
    rng = np.random.default_rng(0)
    assert abs(autocorrelation(rng.normal(size=500))) < 0.15
    assert autocorrelation(np.cumsum(rng.normal(size=500))) > 0.9


def test_interval_coverage():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert interval_coverage(y, y - 1, y + 1) == 1.0
    assert interval_coverage(y, y + 1, y + 2) == 0.0


# -- decoupages ------------------------------------------------------------

def test_blocked_splits_never_train_on_the_future():
    for train, test in blocked_splits(100, 5):
        assert train.stop <= test.start
        assert train.start == 0


def test_expanding_window_grows_and_predicts_forward():
    splits = list(expanding_window_splits(30, min_train=10))
    assert splits[0][0].stop == 10
    assert all(tr.stop == te.start for tr, te in splits)
    assert splits[-1][1].stop == 30


def test_too_few_points_yields_no_split():
    assert list(blocked_splits(5, 5)) == []


# -- references ------------------------------------------------------------

def test_mean_baseline_predicts_the_training_mean(dates):
    y = np.arange(60, dtype=float)
    assert MeanBaseline().fit_predict(dates[:40], y[:40], dates[40:])[0] == pytest.approx(
        y[:40].mean()
    )


def test_persistence_predicts_the_last_value(dates):
    y = np.arange(60, dtype=float)
    assert LastValueBaseline().fit_predict(dates[:40], y[:40], dates[40:])[0] == 39.0


def test_trend_extrapolates_a_linear_series(dates):
    y = np.arange(60, dtype=float)
    pred = TrendBaseline().fit_predict(dates[:40], y[:40], dates[40:])
    assert pred[0] == pytest.approx(40.0, abs=0.5)


def test_ewma_sits_between_mean_and_last_value(dates):
    y = np.arange(60, dtype=float)
    ewma = EwmaBaseline().fit_predict(dates[:40], y[:40], dates[40:])[0]
    assert y[:40].mean() < ewma < 39.0


def test_the_reference_set_is_fixed():
    # Fige pour qu'on ne puisse pas, apres coup, retirer celle qui gagne.
    assert {b.name for b in default_baselines()} == {
        "moyenne", "persistance", "ewma", "tendance"}


# -- validation ------------------------------------------------------------

def _linear(dtr, ytr, dte):
    origin = pd.to_datetime(pd.Series(dtr)).iloc[0]
    t = (pd.to_datetime(pd.Series(dtr)) - origin).dt.days.to_numpy(float)
    a, b = np.polyfit(t, ytr, 1)
    return a * (pd.to_datetime(pd.Series(dte)) - origin).dt.days.to_numpy(float) + b


def test_a_real_signal_is_detected(dates):
    y = np.linspace(0, 1, 60) + np.random.default_rng(0).normal(0, 0.02, 60)
    result = cross_validate("tendance", _linear, dates, y)
    assert result.is_better_than_guessing
    assert result.r2_out_of_sample > 0.9


def test_pure_noise_is_not_rewarded(dates):
    """Le protocole doit refuser de valider un modele sur du bruit."""
    y = np.random.default_rng(1).normal(size=60)
    result = cross_validate("tendance", _linear, dates, y)
    assert not result.is_better_than_guessing


def test_overfitting_gap_is_visible(dates):
    """
    Un modele a beaucoup de parametres obtient un bon ajustement et un
    mauvais score hors echantillon. C'est cet ecart qui constitue le
    resultat, pas le R2 d'ajustement pris isolement.
    """
    y = np.random.default_rng(2).normal(size=60)

    def overfit(dtr, ytr, dte):
        origin = pd.to_datetime(pd.Series(dtr)).iloc[0]
        t = (pd.to_datetime(pd.Series(dtr)) - origin).dt.days.to_numpy(float)
        coef = np.polyfit(t / 100, ytr, min(12, len(ytr) - 1))
        tt = (pd.to_datetime(pd.Series(dte)) - origin).dt.days.to_numpy(float) / 100
        return np.polyval(coef, tt)

    result = cross_validate("surajuste", overfit, dates, y)
    assert result.r2_in_sample > result.r2_out_of_sample
    assert not result.is_better_than_guessing


def test_a_failing_predictor_does_not_crash_the_run(dates):
    def broken(dtr, ytr, dte):
        raise RuntimeError("boum")

    assert cross_validate("casse", broken, dates, np.arange(60.0)).n_folds == 0


def test_compare_ranks_on_out_of_sample_not_on_fit(dates):
    y = np.linspace(0, 1, 60) + np.random.default_rng(3).normal(0, 0.02, 60)
    table = compare({"tendance": _linear,
                     "moyenne": lambda a, b, c: np.full(len(c), np.mean(b))},
                    dates, y)
    assert table.iloc[0]["name"] == "tendance"
    assert list(table["r2_out"]) == sorted(table["r2_out"], reverse=True)
