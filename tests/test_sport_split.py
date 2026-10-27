"""
Decomposition de la charge par discipline.

Pourquoi ce fichier existe : la cible du projet est un rapport de vitesses en
course a pied, et 68,5 % du TRIMP de la fenetre d'evaluation vient du velo.
Un modele a coefficient unique doit trouver un k1 qui vaille a la fois pour
un kilojoule de velo et pour un kilometre de course ; il moyenne un transfert
partiel et un transfert direct, et sort zero. Les tests ci-dessous verrouillent
la decomposition qui permet d'instruire cette explication — et surtout le fait
qu'elle ne change RIEN au chemin historique, dont les six lignes publiees
doivent rester reproductibles au chiffre pres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.report import (
    DURATIONS_S,
    banister_predictor,
    build_variants,
    evaluate_by_duration,
    format_durations,
    load_ablation_pairs,
)
from src.layer_a_signal.envelope import EnvelopeSpec
from src.layer_b_twin.load_model import (
    LOAD_COLUMNS,
    compute_load_curves,
    compute_load_curves_by_sport,
    sport_family,
)
from src.pipeline.orchestrator import build_daily_load


def _sessions(rows):
    """Seances minimales : ce que `build_daily_load` lit reellement."""
    return pd.DataFrame([
        {"session_id": f"s{i}", "date": pd.Timestamp(d), "sport": sport,
         "trimp": trimp, "source": "fit_file"}
        for i, (d, sport, trimp) in enumerate(rows)
    ])


def _daily(n=400, start="2019-01-01", seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D"),
        "trimp_run": rng.gamma(2.0, 20.0, n),
        "trimp_bike": rng.gamma(2.0, 40.0, n),
        "trimp_other": np.zeros(n),
    }).assign(trimp=lambda d: d.trimp_run + d.trimp_bike + d.trimp_other)


# -- taxonomie -------------------------------------------------------------

@pytest.mark.parametrize(("sport", "family"), [
    ("Run", "run"), ("run", "run"), ("Bike", "bike"),
    # VirtualRide porte la meme charge physiologique qu'une sortie velo, meme
    # si sa VITESSE n'est pas comparable : les deux usages divergent, et
    # `envelope_series` la laisse separee pour cette raison.
    ("VirtualRide", "bike"), ("Ride", "bike"),
    ("Swim", "other"), ("Hike", "other"), ("Kayaking", "other"),
    ("Unknown", "other"), ("", "other"),
])
def test_each_discipline_lands_in_its_family(sport, family):
    assert sport_family(sport) == family


def test_a_missing_discipline_is_not_a_crash():
    """Une seance sans sport existe dans le jeu de donnees : 49 « Unknown »."""
    assert sport_family(None) == "other"


# -- charge quotidienne ----------------------------------------------------

def test_the_channels_sum_exactly_to_the_total():
    """
    Le total reste la colonne de reference : les six lignes publiees dans
    docs/results.md en
    dependent. Si les canaux ne sommaient pas au total, l'un des deux
    mentirait.
    """
    daily = build_daily_load(_sessions([
        ("2019-01-01", "Run", 50.0),
        ("2019-01-01", "Bike", 120.0),
        ("2019-01-03", "VirtualRide", 30.0),
        ("2019-01-04", "Swim", 10.0),
    ]))
    channels = daily[list(LOAD_COLUMNS)].sum(axis=1)
    assert np.allclose(channels, daily["trimp"])


def test_virtualride_counts_as_bike_load():
    daily = build_daily_load(_sessions([("2019-01-03", "VirtualRide", 30.0)]))
    row = daily[daily["date"] == "2019-01-03"].iloc[0]
    assert row["trimp_bike"] == pytest.approx(30.0)
    assert row["trimp_run"] == 0.0


def test_an_absent_family_is_zero_and_not_missing():
    """Les consommateurs indexent par nom de colonne : la colonne doit exister."""
    daily = build_daily_load(_sessions([("2019-01-01", "Run", 50.0)]))
    assert list(LOAD_COLUMNS) == [c for c in LOAD_COLUMNS if c in daily.columns]
    assert daily["trimp_bike"].eq(0.0).all()


def test_rest_days_stay_in_the_calendar_with_zeros():
    """Le modele parcourt les jours off pour faire decroitre fitness et fatigue."""
    daily = build_daily_load(_sessions([
        ("2019-01-01", "Run", 50.0), ("2019-01-10", "Bike", 80.0),
    ]))
    assert len(daily) == 10
    assert daily[list(LOAD_COLUMNS)].notna().all().all()


def test_empty_sessions_still_declare_every_column():
    empty = build_daily_load(pd.DataFrame())
    for column in LOAD_COLUMNS:
        assert column in empty.columns


# -- courbes par discipline ------------------------------------------------

def test_a_sport_curve_equals_the_single_sport_curve():
    """
    `compute_load_curves_by_sport` ne doit pas etre un deuxieme calcul : c'est
    `compute_load_curves` applique colonne par colonne.
    """
    daily = _daily()
    by_sport = compute_load_curves_by_sport(daily, 42.0, 12.0, ("trimp_run",))
    alone = compute_load_curves(daily, 42.0, 12.0, trimp_column="trimp_run")
    assert np.allclose(by_sport["fitness_run"], alone.fitness)
    assert np.allclose(by_sport["fatigue_run"], alone.fatigue)


def test_missing_load_columns_are_refused_loudly():
    """Un `daily_load` d'avant la decomposition doit dire quoi faire."""
    old = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=5), "trimp": 1.0})
    with pytest.raises(ValueError, match="cli ingest"):
        compute_load_curves_by_sport(old)


# -- prediteurs ------------------------------------------------------------

def test_the_historical_path_is_numerically_untouched():
    """
    LE test de non-regression. `banister_predictor` a ete generalise ; appele
    sans `load_columns` il doit produire exactement les memes nombres, sinon
    les six lignes publiees dans docs/results.md deviennent fausses sans
    prevenir.
    """
    daily = _daily()
    dates = pd.Series(daily["date"][:300])
    y = np.linspace(0.9, 1.0, 300)
    future = pd.Series(daily["date"][300:350])

    predict = banister_predictor(daily, 42.0, 12.0)
    got = predict(dates, y, future)

    # Reimplementation littérale de la version d'avant la generalisation.
    curves = compute_load_curves(daily, 42.0, 12.0).to_frame()
    f_mean, f_std = curves.fitness.mean(), curves.fitness.std() or 1.0
    g_mean, g_std = curves.fatigue.mean(), curves.fatigue.std() or 1.0
    idx = curves.set_index("date")

    def design(d):
        rows = idx.reindex(pd.to_datetime(pd.Series(d)).dt.normalize())
        return np.column_stack([
            np.ones(len(rows)),
            ((rows.fitness - f_mean) / f_std).to_numpy(),
            -((rows.fatigue - g_mean) / g_std).to_numpy(),
        ])

    X = design(dates)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert np.allclose(got, design(future) @ beta)


def test_one_coefficient_pair_per_requested_sport():
    """Trois disciplines, trois couples : la matrice doit s'elargir."""
    daily = _daily()
    dates = pd.Series(daily["date"][:300])
    y = np.linspace(0.9, 1.0, 300)
    future = pd.Series(daily["date"][300:310])
    # Un ajustement a 7 colonnes sur 300 points reste defini ; c'est sur 85
    # points autocorreles qu'il surapprend, ce que la mesure montre.
    out = banister_predictor(daily, 42.0, 12.0, load_columns=LOAD_COLUMNS)(dates, y, future)
    assert np.isfinite(out).all()


def test_a_zero_load_discipline_is_dropped_not_fitted():
    """
    Une colonne identiquement nulle n'apporte pas une colonne de regression,
    elle la casse. Demander les trois disciplines quand l'une est nulle doit
    donc donner EXACTEMENT le meme ajustement que demander les deux autres.
    """
    daily = _daily()
    assert daily["trimp_other"].eq(0.0).all()
    dates = pd.Series(daily["date"][:300])
    y = np.linspace(0.9, 1.0, 300)
    future = pd.Series(daily["date"][300:310])

    with_zero = banister_predictor(daily, 42.0, 12.0, load_columns=LOAD_COLUMNS)
    without = banister_predictor(daily, 42.0, 12.0,
                                 load_columns=("trimp_run", "trimp_bike"))
    assert np.allclose(with_zero(dates, y, future), without(dates, y, future))


def test_asking_only_for_a_zero_discipline_is_refused():
    daily = _daily()
    with pytest.raises(ValueError, match="Aucune colonne de charge"):
        banister_predictor(daily, 42.0, 12.0, load_columns=("trimp_other",))


def test_every_discipline_zero_is_refused():
    daily = _daily()
    daily[["trimp_run", "trimp_bike", "trimp_other", "trimp"]] = 0.0
    with pytest.raises(ValueError, match="Aucune colonne de charge"):
        banister_predictor(daily, 42.0, 12.0, load_columns=LOAD_COLUMNS)


def test_the_trend_is_a_competing_covariate_not_a_replacement():
    """
    La tendance sert a instruire un confondant : l'athlete passe du velo a la
    course pendant toute la fenetre. Elle doit donc s'AJOUTER a la charge,
    pour qu'on voie laquelle des deux porte le signal.
    """
    daily = _daily()
    dates = pd.Series(daily["date"][:300])
    y = np.linspace(0.9, 1.0, 300)
    future = pd.Series(daily["date"][300:310])
    plain = banister_predictor(daily, 42.0, 12.0)(dates, y, future)
    trend = banister_predictor(daily, 42.0, 12.0, with_trend=True)(dates, y, future)
    assert np.isfinite(trend).all()
    assert not np.allclose(plain, trend)


# -- enregistrement des variantes -----------------------------------------

def test_sport_variants_appear_only_when_the_channels_exist():
    daily = _daily()
    assert "banister par sport t42/12" in build_variants(daily)
    assert "banister course seule t42/12" in build_variants(daily)

    old = daily[["date", "trimp"]]
    names = build_variants(old)
    assert "banister t42/12" in names
    assert not any("par sport" in n for n in names)


def test_every_declared_pair_names_a_registered_variant():
    """
    `load_ablation_pairs` nomme les variantes en dur. Si un nom derive, la
    paire disparait silencieusement du rapport et l'experience n'est plus
    publiee — exactement le genre de panne qui ressemble a un resultat.
    """
    registered = set(build_variants(_daily()))
    for a, b in load_ablation_pairs(42.0, 12.0):
        assert a in registered, a
        assert b in registered, b


# -- protocole par duree ---------------------------------------------------

def _observations(n=120, durations=DURATIONS_S, sparse_at=3600, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="3D")
    frame = pd.DataFrame({"date": dates})
    for d in durations:
        values = 0.9 + 0.05 * rng.random(n)
        if d == sparse_at:
            values[5:] = np.nan          # duree rare, comme 3600 s en vrai
        frame[f"perf_index_{int(d)}"] = values
    return frame


def test_a_duration_with_too_few_observations_is_dropped_and_said_so():
    """
    Sur le jeu de donnees reel la duree 3600 s n'a que neuf observations. Un
    R2 sur neuf points ne veut rien dire : il faut l'ecarter ET le rapporter,
    pas le publier.
    """
    daily = _daily(n=400)
    table, _ = evaluate_by_duration(_observations(), daily, min_train_days=60,
                                    horizon_days=28, min_observations=40)
    dropped = table[(table["duration_s"] == 3600) & (table["variante"] == "ecartee")]
    assert len(dropped) == 1
    assert dropped.iloc[0]["n_obs"] < 40
    assert "ecartee" in format_durations(table)


def test_each_kept_duration_reports_its_own_row_per_variant():
    daily = _daily(n=400)
    table, pairs = evaluate_by_duration(_observations(), daily, min_train_days=60,
                                        horizon_days=28, min_observations=40)
    kept = table[table["variante"] != "ecartee"]
    assert kept["duration_s"].nunique() >= 3
    assert "persistance" in set(kept["variante"])
    # Chaque duree retenue porte sa propre comparaison appariee, avec sa duree.
    assert "duration_s" in pairs.columns
    assert set(pairs["duration_s"]) <= set(kept["duration_s"])


def test_a_duration_absent_from_the_frame_is_simply_skipped():
    daily = _daily(n=400)
    table, _ = evaluate_by_duration(_observations(durations=(300, 600)), daily,
                                    min_train_days=60, min_observations=40)
    assert set(table["duration_s"]) <= {300, 600}


# -- cible velo ------------------------------------------------------------

def _bike_sessions(n=200, start="2018-01-01", seed=2, column="cp20_w"):
    """
    Des seances velo avec une puissance critique, comme le jeu de donnees
    reel : 417 observations de `cp20_w` espacees de deux jours medians.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="2D")
    return pd.DataFrame({
        "session_id": [f"b{i}" for i in range(n)],
        "date": dates,
        "sport": rng.choice(["Bike", "VirtualRide"], n),
        "trimp": rng.gamma(2.0, 40.0, n),
        "source": "fit_file",
        column: 200.0 + 30.0 * rng.random(n),
    })


def test_the_bike_target_is_built_from_power_not_speed():
    """
    Les deux disciplines velo se regroupent pour une PUISSANCE : un watt est
    un watt sur home-trainer comme sur route. Le meme regroupement serait
    faux pour une vitesse, et c'est pour cela qu'il est demande explicitement.
    """
    from src.cli import BIKE_SPORTS, BIKE_TARGET_COLUMN, _evaluate_bike_target

    sessions = _bike_sessions()
    assert BIKE_TARGET_COLUMN in sessions.columns
    assert set(BIKE_SPORTS) == {"bike", "virtualride"}
    results, floors = _evaluate_bike_target(
        sessions, _daily(n=500, start="2018-01-01"), (42.0, 12.0), 28, 120,
        EnvelopeSpec("pi", 42, min_periods=3))
    assert results is not None
    assert "banister velo seul t42/12" in results
    assert results["banister velo seul t42/12"].n_folds > 0
    assert floors is not None and not floors.empty


def test_no_bike_power_means_no_bike_protocol():
    """Un athlete sans capteur de puissance ne doit pas faire planter la commande."""
    from src.cli import _evaluate_bike_target

    sessions = _bike_sessions().drop(columns=["cp20_w"])
    results, floors = _evaluate_bike_target(
        sessions, _daily(n=500), (42.0, 12.0), 28, 120,
        EnvelopeSpec("pi", 42, min_periods=3))
    assert results is None and floors is None


def test_too_few_bike_observations_publish_nothing():
    """
    Une variante qui n'a presque rien execute ne publie rien : c'est la regle
    du depot, et elle vaut aussi pour un protocole entier.
    """
    from src.cli import _evaluate_bike_target

    results, _ = _evaluate_bike_target(
        _bike_sessions(n=10), _daily(n=500), (42.0, 12.0), 28, 120,
        EnvelopeSpec("pi", 42, min_periods=3))
    assert results is None


def test_running_load_is_registered_separately_from_bike_load():
    """
    Les deux variantes specifiques doivent coexister : c'est leur comparaison
    sur une cible velo qui sert de controle de validite discriminante.
    """
    names = build_variants(_daily())
    assert "banister course seule t42/12" in names
    assert "banister velo seul t42/12" in names
