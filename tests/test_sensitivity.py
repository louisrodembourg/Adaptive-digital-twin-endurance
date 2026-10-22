"""
Plancher de detectabilite.

Le depot conclut a l'absence d'effet de la charge. Tant qu'on ne sait pas
quel effet le protocole SAIT voir, « on n'a rien trouve » et « il n'y a rien »
sont indistinguables — et le second ne se deduit pas du premier. Ces tests
verrouillent la mecanique qui rend la conclusion falsifiable, ainsi que la
garantie d'honnetete qui va avec : rien de ce qui sort d'ici ne peut etre
confondu avec une mesure de l'athlete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.sensitivity import (
    DEFAULT_DELTAS,
    Floor,
    detectability_curve,
    detection_floor,
    format_detectability,
    injected_signal,
)


def _daily(n=900, start="2018-01-01", seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D"),
        "trimp_run": rng.gamma(2.0, 20.0, n),
        "trimp_bike": rng.gamma(2.0, 40.0, n),
        "trimp_other": np.zeros(n),
    }).assign(trimp=lambda d: d.trimp_run + d.trimp_bike)


def _dense(daily, seed=0):
    """
    Une cible qui ressemble a l'enveloppe reelle : un maximum glissant, donc
    fortement autocorrelee. Sans cette propriete la persistance serait une
    reference faible et le plancher mesure n'aurait rien a voir avec celui
    que le depot publie.
    """
    rng = np.random.default_rng(seed)
    noisy = pd.Series(0.9 + 0.04 * rng.random(len(daily)))
    return pd.DataFrame({
        "date": daily["date"],
        "perf": noisy.rolling(42, min_periods=1).max().to_numpy(),
    })


# -- le signal injecte -----------------------------------------------------

def test_the_injected_signal_is_standardised():
    """
    `delta` s'exprime en ecarts-types d'enveloppe. Cela n'a de sens que si le
    signal multiplie est lui-meme reduit — sinon le nombre publie dependrait
    de l'unite du TRIMP.
    """
    daily = _daily()
    signal = injected_signal(daily, daily["date"])
    assert signal.mean() == pytest.approx(0.0, abs=1e-9)
    assert signal.std() == pytest.approx(1.0, abs=1e-9)


def test_the_signal_has_the_shape_the_model_looks_for():
    """
    On donne au modele exactement ce qu'il cherche : fitness - fatigue aux
    memes constantes de temps. C'est ce qui rend le plancher OPTIMISTE, et
    donc une borne inferieure sur ce qu'il faudrait pour etre vu.
    """
    daily = _daily()
    from src.layer_b_twin.load_model import compute_load_curves_by_sport
    curves = compute_load_curves_by_sport(daily, 42.0, 12.0, ("trimp_run",))
    expected = curves["fitness_run"] - curves["fatigue_run"]
    got = injected_signal(daily, daily["date"], 42.0, 12.0)
    # Egales a une standardisation affine pres : la correlation vaut 1.
    assert np.corrcoef(expected, got)[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_running_load_is_the_default_channel():
    """
    C'est la seule charge dont on puisse supposer qu'elle agisse sur une
    vitesse en course a pied ; le protocole doit etre teste sur sa capacite a
    voir celle-la, pas la charge totale aux deux tiers cycliste.
    """
    daily = _daily()
    run = injected_signal(daily, daily["date"], column="trimp_run")
    bike = injected_signal(daily, daily["date"], column="trimp_bike")
    assert not np.allclose(run, bike)
    assert np.allclose(run, injected_signal(daily, daily["date"]))


def test_an_empty_load_channel_is_refused():
    daily = _daily()
    daily["trimp_other"] = 0.0
    with pytest.raises(ValueError, match="Signal de charge vide"):
        injected_signal(daily, pd.Series([], dtype="datetime64[ns]"),
                        column="trimp_other")


# -- la courbe -------------------------------------------------------------

@pytest.fixture(scope="module")
def curve():
    daily = _daily()
    return detectability_curve(_dense(daily), daily, min_train_days=200,
                              horizon_days=28)


def test_a_bigger_injected_effect_is_better_recovered(curve):
    """
    La propriete sans laquelle le plancher ne voudrait rien dire : le modele
    doit retrouver d'autant mieux l'effet qu'il est gros. Si la courbe
    n'etait pas croissante, c'est la mecanique qui serait cassee, pas le
    protocole qui serait aveugle.
    """
    against_mean = curve[curve["reference"] == "moyenne"].sort_values("delta")
    r2 = against_mean["r2_modele"].to_numpy()
    assert np.all(np.diff(r2) > 0)


def test_no_effect_is_not_detected_and_a_huge_one_is(curve):
    at_zero = curve[(curve["delta"] == 0.0) & (curve["reference"] == "moyenne")]
    at_max = curve[(curve["delta"] == max(DEFAULT_DELTAS))
                   & (curve["reference"] == "moyenne")]
    assert not bool(at_zero.iloc[0]["detecte"])
    assert bool(at_max.iloc[0]["detecte"])


def test_detection_requires_the_interval_to_exclude_zero(curve):
    """
    Un P(gain) eleve ne suffit pas : c'est la regle du depot, et le plancher
    doit l'appliquer comme le reste du rapport.
    """
    for _, row in curve.iterrows():
        assert bool(row["detecte"]) == bool(row["ci_low"] > 0.0)


def test_the_floor_is_the_smallest_detected_amplitude(curve):
    floor = detection_floor(curve, "moyenne")
    assert floor.found
    detected = curve[(curve["reference"] == "moyenne") & curve["detecte"]]
    assert floor.delta == detected["delta"].min()


def test_persistence_is_a_harder_reference_than_the_mean(curve):
    """
    Sur une cible autocorrelee, battre la persistance demande un effet plus
    gros que battre la moyenne. C'est precisement ce qui prive de puissance
    la comparaison que le depot publie en premier.
    """
    mean_floor = detection_floor(curve, "moyenne")
    persistence_floor = detection_floor(curve, "persistance")
    if mean_floor.found and persistence_floor.found:
        assert persistence_floor.delta >= mean_floor.delta
    else:
        assert mean_floor.found


def test_an_undetected_floor_says_so_rather_than_claiming_zero():
    """
    Un plancher introuvable est un resultat : il signifie que le protocole ne
    verrait meme pas l'effet le plus gros teste. Le rendre comme `None` et
    non comme la derniere amplitude evite de publier une borne inventee.
    """
    floor = Floor(reference="persistance", delta=None, deltas_tested=(0.0, 1.0))
    assert not floor.found
    assert "aucun plancher" in floor.describe()
    assert "1.00" in floor.describe()


def test_nothing_leaves_this_module_that_could_pass_for_a_measurement(curve):
    """
    La garantie d'honnetete. L'enveloppe perturbee n'existe qu'en memoire :
    l'interface ne rend que des statistiques, jamais une serie qu'on pourrait
    ecrire dans `data/` et confondre plus tard avec une mesure.
    """
    assert set(curve.columns) == {
        "delta", "reference", "r2_modele", "r2_reference", "skill",
        "ci_low", "ci_high", "p_a_better", "n_blocks", "detecte",
    }
    assert len(curve) == len(curve.drop_duplicates(["delta", "reference"]))


def test_the_report_names_every_floor(curve):
    text = format_detectability(curve)
    assert "moyenne" in text and "persistance" in text
    assert "ecart-type" in text


def test_an_empty_curve_does_not_pretend_to_have_a_floor():
    assert "Aucune amplitude" in format_detectability(pd.DataFrame())


# -- le plancher atteint des les donnees reelles ---------------------------

def test_an_effect_seen_without_injection_is_said_differently():
    """
    Un plancher a delta = 0 ne dit pas que le protocole est sensible : il dit
    qu'il a trouve quelque chose dans les donnees REELLES. C'est le cas de la
    cible velo contre « predire la moyenne », et le confondre avec un
    plancher ordinaire inverserait la lecture.
    """
    floor = Floor(reference="moyenne", delta=0.0, deltas_tested=(0.0, 1.0))
    assert floor.found
    assert floor.detected_without_injection
    assert "SANS injection" in floor.describe()


def test_an_ordinary_floor_is_not_mistaken_for_a_real_effect():
    floor = Floor(reference="persistance", delta=1.0, deltas_tested=(0.0, 1.0))
    assert floor.found
    assert not floor.detected_without_injection
    assert "SANS injection" not in floor.describe()
    assert "1.00" in floor.describe()


def test_the_tested_model_consumes_the_injected_discipline():
    """
    Mesurer le plancher d'un modele nourri de course quand l'effet est
    injecte depuis le velo ne dirait rien du protocole : la variante par
    defaut suit donc la discipline injectee.
    """
    daily = _daily()
    dense = _dense(daily)
    run = detectability_curve(dense, daily, deltas=(0.0,), column="trimp_run",
                              min_train_days=200)
    bike = detectability_curve(dense, daily, deltas=(0.0,), column="trimp_bike",
                               min_train_days=200)
    assert not run.empty and not bike.empty
    # Deux modeles differents sur la meme cible : les scores doivent differer.
    assert not np.allclose(run["r2_modele"].to_numpy(), bike["r2_modele"].to_numpy())


def test_an_unknown_discipline_demands_an_explicit_variant():
    daily = _daily()
    with pytest.raises(ValueError, match=r"[Pp]asser `variant`"):
        detectability_curve(_dense(daily), daily, deltas=(0.0,),
                            column="trimp_other", min_train_days=200)
