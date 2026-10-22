"""
Enveloppe de performance.

Le projet mesurait la performance par le meilleur effort de chaque seance,
qui decrit surtout ce qu'etait la seance plutot que la capacite de
l'athlete. Sur les donnees reelles cette serie a une autocorrelation de
0,15 au velo et de -0,12 en course : du bruit autour d'un niveau, que rien
ne peut predire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.layer_a_signal.envelope import (
    EnvelopeSpec,
    describe_predictability,
    envelope_series,
    rolling_envelope,
)


def _observations(values, start="2019-01-01", freq="3D"):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(values), freq=freq),
        "v": np.asarray(values, dtype=float),
    })


# -- construction ----------------------------------------------------------

def test_envelope_is_the_running_maximum_over_the_window():
    obs = _observations([1.0, 5.0, 2.0, 3.0, 4.0], freq="1D")
    env = rolling_envelope(obs, EnvelopeSpec("v", window_days=365, min_periods=1))
    assert env["v_env365"].tolist() == [1.0, 5.0, 5.0, 5.0, 5.0]


def test_envelope_decays_when_the_window_moves_past_a_peak():
    """
    La fenetre est en JOURS, pas en nombre de seances : une periode creuse
    doit faire retomber l'enveloppe, ce qu'un decompte de lignes ne ferait
    pas.
    """
    obs = pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2019-01-05", "2019-06-01"]),
        "v": [9.0, 4.0, 5.0],
    })
    env = rolling_envelope(obs, EnvelopeSpec("v", window_days=30, min_periods=1))
    assert env["v_env30"].iloc[0] == 9.0
    assert env["v_env30"].iloc[-1] == 5.0  # le pic de janvier est sorti


def test_envelope_never_looks_ahead():
    obs = _observations([1.0, 1.0, 1.0, 9.0], freq="1D")
    env = rolling_envelope(obs, EnvelopeSpec("v", window_days=365, min_periods=1))
    assert env["v_env365"].iloc[0] == 1.0  # le 9 arrive plus tard


def test_several_sessions_the_same_day_keep_the_best():
    obs = pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2019-01-01", "2019-01-02"]),
        "v": [3.0, 7.0, 4.0],
    })
    env = rolling_envelope(obs, EnvelopeSpec("v", window_days=30, min_periods=1))
    assert env["v_env30"].iloc[0] == 7.0


def test_thin_windows_are_dropped():
    # Une enveloppe adossee a une seule seance n'est qu'une seance.
    obs = _observations([1.0, 2.0, 3.0, 4.0], freq="1D")
    env = rolling_envelope(obs, EnvelopeSpec("v", window_days=30, min_periods=3))
    assert len(env) == 2


def test_missing_values_are_ignored():
    obs = _observations([1.0, np.nan, 3.0], freq="1D")
    env = rolling_envelope(obs, EnvelopeSpec("v", window_days=30, min_periods=1))
    assert len(env) == 2


@pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame({"date": [], "autre": []})])
def test_empty_or_wrong_column_is_not_an_error(frame):
    assert rolling_envelope(frame, EnvelopeSpec("v")).empty


# -- la raison d'etre du module -------------------------------------------

def test_the_envelope_is_far_more_predictable_than_raw_efforts():
    """
    C'est tout l'argument : le meilleur effort d'une seance est du bruit,
    l'enveloppe est une grandeur qui bouge lentement. Mesure sur les donnees
    reelles : 0,15 contre 0,93 au velo.
    """
    rng = np.random.default_rng(0)
    obs = _observations(rng.normal(4.0, 0.4, 120))
    env = rolling_envelope(obs, EnvelopeSpec("v", window_days=42, min_periods=3))

    raw_ac = describe_predictability(obs["v"])["autocorr_lag1"]
    env_ac = describe_predictability(env["v_env42"])["autocorr_lag1"]
    assert env_ac > 0.8
    assert env_ac > raw_ac + 0.4


def test_envelope_varies_less_than_the_raw_series():
    rng = np.random.default_rng(1)
    obs = _observations(rng.normal(300.0, 30.0, 120))
    env = rolling_envelope(obs, EnvelopeSpec("v", window_days=42, min_periods=3))
    assert env["v_env42"].std() < obs["v"].std()


def test_describe_flags_white_noise():
    rng = np.random.default_rng(2)
    assert abs(describe_predictability(rng.normal(size=400))["autocorr_lag1"]) < 0.15


# -- integration avec sessions.parquet -------------------------------------

def _sessions():
    rng = np.random.default_rng(3)
    n = 80
    return pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=n, freq="4D"),
        "sport": ["Bike" if i % 3 else "Run" for i in range(n)],
        "cp20_w": rng.normal(280, 25, n),
    })


def test_sport_filter_selects_one_discipline():
    env = envelope_series(_sessions(), "cp20_w", sport_filter="bike")
    assert not env.empty
    assert len(env) < len(_sessions())


def test_virtualride_is_not_swept_up_with_bike():
    """
    `startswith("bike")` ne doit pas ramasser VirtualRide : ce sont deux
    disciplines distinctes, et les confondre serait une decision invisible.
    """
    sessions = pd.DataFrame({
        "date": pd.date_range("2019-01-01", periods=6, freq="1D"),
        "sport": ["Bike", "VirtualRide"] * 3,
        "cp20_w": [300.0, 999.0] * 3,
    })
    env = envelope_series(sessions, "cp20_w", sport_filter="bike", min_periods=1)
    assert env["cp20_w_env42"].max() == 300.0


# -- grille journaliere ----------------------------------------------------

def test_daily_calendar_is_the_same_quantity_sampled_more_finely():
    """
    Le garde-fou de la grille journaliere. Un maximum glissant sur 42 jours
    est defini chaque jour a partir des seules mesures reelles de la
    fenetre : passer a une grille journaliere ne doit donc rien changer aux
    valeurs des jours deja presents. Si ce test tombe, la grille fabrique de
    la donnee au lieu de la reechantillonner.
    """
    rng = np.random.default_rng(4)
    obs = _observations(rng.normal(4.0, 0.4, 60), freq="5D")
    spec = EnvelopeSpec("v", window_days=42, min_periods=3)

    sparse = rolling_envelope(obs, spec)
    dense = rolling_envelope(obs, spec, calendar="daily")

    aligned = dense[dense["date"].isin(sparse["date"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(aligned, sparse)


def test_daily_calendar_fills_every_day_between_the_first_and_last_measure():
    obs = pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-20"]),
        "v": [1.0, 2.0, 3.0, 4.0],
    })
    dense = rolling_envelope(obs, EnvelopeSpec("v", window_days=42, min_periods=3),
                             calendar="daily")
    # Du 3 janvier (3e mesure, donc min_periods atteint) au 20 janvier.
    assert dense["date"].tolist() == pd.date_range("2019-01-03", "2019-01-20").tolist()
    assert dense["n_in_window"].min() >= 3


def test_daily_calendar_still_decays_once_the_window_empties():
    """
    Une grille journaliere ne doit pas prolonger la derniere valeur connue :
    les jours dont la fenetre ne contient plus assez de mesures sortent, au
    lieu d'etre remplis par report.
    """
    obs = pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2019-01-02", "2019-01-03", "2019-06-01"]),
        "v": [1.0, 2.0, 9.0, 4.0],
    })
    dense = rolling_envelope(obs, EnvelopeSpec("v", window_days=30, min_periods=3),
                             calendar="daily")
    # La fenetre pandas est ouverte a gauche : au 31 janvier elle ne couvre
    # plus que les 2 et 3 janvier, donc moins de min_periods mesures.
    assert dense["date"].max() == pd.Timestamp("2019-01-30")
    assert pd.Timestamp("2019-06-01") not in set(dense["date"])


def test_unknown_calendar_is_refused():
    obs = _observations([1.0, 2.0, 3.0], freq="1D")
    with pytest.raises(ValueError, match="calendar"):
        rolling_envelope(obs, EnvelopeSpec("v"), calendar="weekly")


# -- regroupement de disciplines -------------------------------------------

def _mixed_sports():
    """Trois disciplines le meme jour, pour distinguer les filtres."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2019-01-05", "2019-01-09",
                                "2019-01-13", "2019-01-17", "2019-01-21"]),
        "sport": ["Bike", "VirtualRide", "Run", "Bike", "VirtualRide", "Run"],
        "cp20_w": [200.0, 210.0, 999.0, 220.0, 230.0, 999.0],
    })


def test_a_single_prefix_still_excludes_virtualride():
    """
    Le comportement historique ne doit pas bouger : "bike" retient Bike et
    pas VirtualRide, parce qu'une VITESSE sur home-trainer n'est pas
    comparable a une vitesse sur route.
    """
    out = envelope_series(_mixed_sports(), "cp20_w", window_days=42,
                          sport_filter="bike", min_periods=1)
    assert out["cp20_w_env42"].max() == 220.0


def test_a_tuple_of_prefixes_groups_the_disciplines():
    """
    Un watt est un watt : pour une PUISSANCE les deux disciplines velo se
    regroupent, ce qui serait faux pour une vitesse. D'ou le tuple explicite.
    """
    out = envelope_series(_mixed_sports(), "cp20_w", window_days=42,
                          sport_filter=("bike", "virtualride"), min_periods=1)
    assert out["cp20_w_env42"].max() == 230.0


def test_the_filter_keeps_running_out_of_a_bike_envelope():
    out = envelope_series(_mixed_sports(), "cp20_w", window_days=42,
                          sport_filter=("bike", "virtualride"), min_periods=1)
    # 999 est la valeur sentinelle posee sur les lignes Run.
    assert 999.0 not in set(out["cp20_w_env42"])


def test_envelope_series_can_ask_for_the_daily_grid():
    sparse = envelope_series(_mixed_sports(), "cp20_w", window_days=42,
                             sport_filter=("bike", "virtualride"), min_periods=1)
    dense = envelope_series(_mixed_sports(), "cp20_w", window_days=42,
                            sport_filter=("bike", "virtualride"), min_periods=1,
                            calendar="daily")
    assert len(dense) > len(sparse)
    aligned = dense[dense["date"].isin(sparse["date"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(aligned, sparse)
