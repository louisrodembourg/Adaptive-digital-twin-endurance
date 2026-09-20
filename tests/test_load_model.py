"""
Modele de charge : le defaut central du depot.

compute_fitness_fatigue omettait le facteur (1 - exp(-1/tau)), transformant
une moyenne mobile exponentielle en somme cumulee amortie. Elle atteignait
4871 chez l'athlete du depot, et la performance en sortait a 941,57 la ou
tous les consommateurs attendaient 1,0 : le temps estime sur semi-marathon
s'affichait a cinq secondes et la probabilite de succes valait exactement 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.layer_b_twin.load_model import (
    acute_chronic_ratio,
    compute_load_curves,
    exponential_load,
    form_thresholds,
)


def _daily(trimps, start="2019-01-01"):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(trimps), freq="D"),
        "trimp": trimps,
    })


# -- la correction ---------------------------------------------------------

def test_constant_load_converges_to_that_load():
    """
    C'est tout le bug en une assertion : une charge constante doit donner une
    fitness egale a cette charge, pas tau fois plus grande.
    """
    assert exponential_load(np.full(400, 50.0), 42.0)[-1] == pytest.approx(50.0, abs=0.5)


def test_output_stays_in_trimp_units():
    fitness = exponential_load(np.full(400, 50.0), 42.0)
    # La somme cumulee non normalisee valait environ tau fois la charge.
    assert fitness.max() < 60.0


def test_a_longer_time_constant_reacts_more_slowly():
    load = np.concatenate([np.zeros(50), np.full(50, 100.0)])
    slow = exponential_load(load, 42.0)
    fast = exponential_load(load, 12.0)
    assert fast[-1] > slow[-1]          # la fatigue monte plus vite
    assert fast[60] > slow[60]


def test_load_decays_after_training_stops():
    load = np.concatenate([np.full(100, 80.0), np.zeros(100)])
    curve = exponential_load(load, 42.0)
    assert curve[-1] < curve[99] * 0.2


def test_non_positive_time_constant_is_rejected():
    with pytest.raises(ValueError):
        exponential_load(np.ones(10), 0.0)


def test_empty_history_is_not_an_error():
    assert exponential_load(np.array([]), 42.0).size == 0
    assert compute_load_curves(pd.DataFrame()).fitness.size == 0


# -- courbes ---------------------------------------------------------------

def test_fitness_must_be_slower_than_fatigue():
    """Sans cet ordre les deux courbes sont interchangeables."""
    with pytest.raises(ValueError, match="fatigue"):
        compute_load_curves(_daily([50.0] * 100), tau1=10.0, tau2=42.0)


def test_form_lags_by_one_day():
    # La charge du jour n'a pas encore produit son effet au moment de decider
    # de s'entrainer.
    curves = compute_load_curves(_daily([0.0] + [100.0] * 60))
    assert curves.form[0] == 0.0
    assert curves.form[2] == pytest.approx(curves.fitness[1] - curves.fatigue[1])


def test_hard_block_drives_form_negative():
    curves = compute_load_curves(_daily([20.0] * 60 + [150.0] * 20))
    assert curves.form[-1] < 0


def test_taper_drives_form_positive():
    curves = compute_load_curves(_daily([100.0] * 120 + [10.0] * 20))
    assert curves.form[-1] > 0


# -- seuils ----------------------------------------------------------------

def test_thresholds_are_derived_not_borrowed():
    """
    -30 et +25 sont des conventions TrainingPeaks pour le TSS, pas pour le
    TRIMP : les emprunter ajouterait un troisieme nombre magique.
    """
    rng = np.random.default_rng(0)
    curves = compute_load_curves(_daily(rng.gamma(2, 25, 500)))
    low, high = form_thresholds(curves.form)
    assert low < 0 < high


def test_short_history_refuses_to_produce_thresholds():
    with pytest.raises(ValueError, match="trop court"):
        form_thresholds(np.zeros(10))


# -- ACWR ------------------------------------------------------------------

def test_acwr_is_one_on_steady_training():
    acwr = acute_chronic_ratio(_daily([50.0] * 60)).dropna()
    assert acwr.iloc[-1] == pytest.approx(1.0, abs=0.01)


def test_acwr_rises_on_a_sudden_block():
    acwr = acute_chronic_ratio(_daily([30.0] * 40 + [120.0] * 10)).dropna()
    assert acwr.iloc[-1] > 1.5


def test_acwr_is_undefined_before_the_chronic_window():
    assert acute_chronic_ratio(_daily([50.0] * 20)).isna().all()
