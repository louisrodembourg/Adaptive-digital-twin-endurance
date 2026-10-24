"""
Modele hierarchique : ce que treize ajustements separes ne pouvaient pas dire.

Rejouer le protocole athlete par athlete compte des verdicts ; il n'estime pas
une dispersion. Un modele hierarchique rend `sigma_k_net`, l'ecart entre
athletes, et partage les constantes de temps pour qu'elles soient contraintes
par quinze historiques au lieu d'un.

Ces tests couvrent la preparation des donnees et la structure du modele, sans
echantillonner : un ajustement reel coute une heure et vit derriere le
marqueur `mcmc`, comme le reste du chemin bayesien du depot. Ce qui est
verrouille ici est ce qui rend l'ajustement interpretable — la reduction des
cibles, la parametrisation non centree, et la presence du terme de tendance
concurrent sans lequel une constante de fitness qui fuit transforme la charge
en tendance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.layer_b_twin.hierarchical import (
    DEFAULT_STRIDE_DAYS,
    MIN_POINTS,
    AthleteBlock,
    CohortData,
    build_hierarchical_model,
    prepare_athlete,
)


def _daily(n=900, start="2023-01-01", seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D"),
        "trimp_bike": rng.gamma(2.0, 30.0, n),
    })
    return frame.assign(trimp=frame.trimp_bike)


def _power(daily, n=200, step_days=4, level=250.0, seed=1):
    rng = np.random.default_rng(seed)
    dates = daily["date"].iloc[::step_days][:n]
    return pd.DataFrame({"date": dates.to_numpy(),
                         "pi": level + rng.normal(0, 8.0, len(dates))})


def _block(uuid="a", n_days=400, n_points=30, seed=0):
    rng = np.random.default_rng(seed)
    return AthleteBlock(
        uuid=uuid,
        trimp=rng.gamma(2.0, 30.0, n_days),
        obs_days=np.linspace(0, n_days - 1, n_points).astype(int),
        y=rng.normal(0, 1, n_points),
        trend=np.linspace(-1, 1, n_points),
    )


# ── preparation ──────────────────────────────────────────────────────────

def test_the_target_is_standardised_so_gains_can_be_pooled():
    """
    Les enveloppes vont de 170 a 260 W selon l'athlete. Mettre en commun un
    gain n'a de sens que si la grandeur expliquee a la meme echelle partout ;
    sinon `sigma_k_net` mesurerait des unites, pas de l'heterogeneite.
    """
    daily = _daily()
    low = prepare_athlete("bas", daily, _power(daily, level=180.0))
    high = prepare_athlete("haut", daily, _power(daily, level=320.0))
    for block in (low, high):
        assert block is not None
        assert abs(float(block.y.mean())) < 1e-9
        assert float(block.y.std()) == pytest.approx(1.0, abs=1e-9)


def test_one_point_every_stride_days():
    """
    L'enveloppe est un maximum glissant sur 42 jours : deux jours consecutifs
    sont quasi identiques, et le noyau coute n_points x n_jours. On lit la
    meme grandeur moins souvent, on ne la degrade pas.
    """
    daily = _daily()
    dense = prepare_athlete("a", daily, _power(daily), stride_days=1)
    sparse = prepare_athlete("a", daily, _power(daily),
                             stride_days=DEFAULT_STRIDE_DAYS)
    assert dense.n_points > sparse.n_points
    assert sparse.n_points == pytest.approx(dense.n_points / DEFAULT_STRIDE_DAYS,
                                            rel=0.2)


def test_the_envelope_label_follows_the_window():
    """
    L'etiquette de la colonne est `pi_env{window}` : la coder en dur a 42
    cassait toute autre fenetre en silence.
    """
    daily = _daily()
    for window in (28, 42, 56):
        block = prepare_athlete("a", daily, _power(daily), window_days=window)
        assert block is not None and block.n_points > 0


def test_an_athlete_with_too_few_points_is_left_out():
    """En dessous du seuil, sa ligne n'apprend rien et ajoute des parametres."""
    daily = _daily(n=200)
    assert prepare_athlete("a", daily, _power(daily, n=3, step_days=4)) is None


@pytest.mark.parametrize(("daily", "power"), [
    (pd.DataFrame(), None),
    (None, pd.DataFrame()),
])
def test_missing_data_yields_no_block_rather_than_an_error(daily, power):
    base = _daily()
    d = base if daily is None else daily
    p = _power(base) if power is None else power
    assert prepare_athlete("a", d, p) is None


def test_a_load_channel_that_is_absent_yields_no_block():
    daily = _daily()[["date", "trimp"]]
    assert prepare_athlete("a", daily, _power(_daily())) is None


def test_the_trend_is_standardised_and_increasing():
    """
    La tendance doit etre reduite comme la cible, pour que son coefficient
    soit comparable a celui de la charge — c'est une concurrence, pas un
    ajout decoratif.
    """
    daily = _daily()
    block = prepare_athlete("a", daily, _power(daily))
    assert block.trend.shape == block.y.shape
    assert abs(float(block.trend.mean())) < 1e-9
    assert float(block.trend.std()) == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(block.trend) >= 0)


# ── la cohorte ───────────────────────────────────────────────────────────

def test_the_cohort_reports_what_it_holds():
    data = CohortData((_block("aaa", seed=0), _block("bbb", seed=1)))
    assert data.n_athletes == 2
    assert data.n_points == sum(b.n_points for b in data.blocks)
    assert "2 athlete(s)" in data.describe()


def test_an_empty_cohort_cannot_be_fitted():
    pytest.importorskip("pymc")
    with pytest.raises(ValueError, match="Aucun athlete"):
        build_hierarchical_model(CohortData(()))


# ── structure du modele ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    pytest.importorskip("pymc")
    return build_hierarchical_model(
        CohortData((_block("aaa", seed=0), _block("bbb", seed=1),
                    _block("ccc", seed=2))))


def test_the_time_constants_are_shared_and_ordered(model):
    """
    tau1 et tau2 ne sont pas identifiees sur un athlete — le posterieur du
    depot ne les resserre que de 41 %. Les partager est la raison d'etre de
    ce modele, et l'ordre est impose par construction pour supprimer la
    permutation d'etiquettes entre chaines.
    """
    names = {v.name for v in model.free_RVs}
    assert "tau2" in names and "delta" in names
    assert "tau1" not in names          # deterministe : tau2 + delta
    deterministic = {v.name for v in model.deterministics}
    assert "tau1" in deterministic


def test_gains_are_per_athlete_and_time_constants_are_not(model):
    """
    Un gain par athlete, un seul couple de constantes de temps. La forme
    symbolique de PyTensor vaut (None,) avant evaluation : ce sont les `dims`
    qui portent l'intention, et c'est elles qu'il faut verifier.
    """
    dims = model.named_vars_to_dims
    assert dims.get("z_net") == ("athlete",)
    assert dims.get("z_fat") == ("athlete",)
    assert dims.get("p0") == ("athlete",)
    assert "tau2" not in dims          # partagee : aucune dimension athlete
    assert len(model.coords["athlete"]) == 3


def test_the_parametrisation_is_non_centred(model):
    """
    La forme centree place un entonnoir entre `sigma` et les gains, que NUTS
    traverse mal. C'est le defaut classique des modeles hierarchiques, et il
    se corrige par la parametrisation et non par plus d'iterations.
    """
    names = {v.name for v in model.free_RVs}
    assert {"z_net", "z_fat", "z_trend"} <= names
    deterministic = {v.name for v in model.deterministics}
    assert {"k_net", "k2", "k1", "k_trend"} <= deterministic


def test_the_net_effect_is_sampled_and_not_each_gain(model):
    """
    Meme raison que dans le modele mono-athlete : k1 et k2 ressortent
    correles a 0,98, seul leur effet net est contraint, et les echantillonner
    separement avait coute vingt-deux heures.
    """
    names = {v.name for v in model.free_RVs}
    assert "z_net" in names
    assert "k1" not in names
    assert "k1" in {v.name for v in model.deterministics}


def test_a_competing_trend_term_is_present(model):
    """
    Sans lui, une constante de fitness qui fuit vers des centaines de jours
    transforme la charge cumulee en tendance, et le gain mesure l'une en
    croyant mesurer l'autre. Mesure sur la cohorte : tau1 partait a 127 jours.
    """
    assert "z_trend" in {v.name for v in model.free_RVs}
    assert "k_trend" in {v.name for v in model.deterministics}


def test_every_athlete_has_its_own_likelihood(model):
    observed = {v.name for v in model.observed_RVs}
    assert observed == {"perf_0", "perf_1", "perf_2"}


def test_the_model_carries_its_data_for_later_diagnostics(model):
    assert model.cohort_data.n_athletes == 3


def test_the_inclusion_threshold_is_declared():
    assert MIN_POINTS >= 3


def test_a_trace_that_predates_the_cohort_says_so():
    """
    Le cache grandit des qu'on telecharge des archives. Rejouer une analyse
    sur une trace plus ancienne donnait un `KeyError` d'xarray illisible ;
    il faut nommer la cause et la commande qui la corrige.
    """
    pytest.importorskip("arviz")
    import xarray as xr

    from src.layer_b_twin.hierarchical import per_athlete_effects

    # Une trace qui ne connait que deux athletes.
    k_net = xr.DataArray(
        np.zeros((1, 10, 2)),
        dims=("chain", "draw", "athlete"),
        coords={"chain": [0], "draw": range(10), "athlete": ["aaa", "bbb"]},
    )
    trace = type("Trace", (), {"posterior": xr.Dataset({"k_net": k_net})})()

    # Une cohorte qui en compte trois.
    data = CohortData((_block("aaa"), _block("bbb"), _block("ccc")))
    with pytest.raises(ValueError, match="cli pool"):
        per_athlete_effects(trace, data)


def test_a_matching_trace_is_read_without_complaint():
    pytest.importorskip("arviz")
    import xarray as xr

    from src.layer_b_twin.hierarchical import per_athlete_effects

    k_net = xr.DataArray(
        np.full((2, 50, 2), 0.4),
        dims=("chain", "draw", "athlete"),
        coords={"chain": [0, 1], "draw": range(50), "athlete": ["aaa", "bbb"]},
    )
    trace = type("Trace", (), {"posterior": xr.Dataset({"k_net": k_net})})()
    data = CohortData((_block("aaa"), _block("bbb")))
    effects = per_athlete_effects(trace, data)
    assert list(effects["athlete"]) == ["aaa", "bbb"]
    assert effects["k_net_moyen"].iloc[0] == pytest.approx(0.4)
