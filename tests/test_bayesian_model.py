"""
Modele bayesien.

Ce chemin n'avait jamais ete execute : il appelait `pt.scan`, symbole qui
n'existe pas. Quatre autres defauts l'auraient fait echouer au premier
echantillonnage, dont un corps de recurrence mathematiquement faux —
`fit_prev * decay + trimp_t`, sans le facteur (1 - e^{-1/tau}) — c'est-a-dire
le defaut deja corrige dans load_model.py, duplique ici.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.layer_b_twin.bayesian_model import (
    HAS_PYMC,
    NotFittedError,
    build_banister_model,
    load_kernel,
)
from src.layer_b_twin.load_model import exponential_load

pytestmark = pytest.mark.filterwarnings("ignore")


# -- noyau de convolution --------------------------------------------------

@pytest.mark.parametrize("tau", [7.0, 12.0, 42.0, 90.0])
def test_kernel_reproduces_the_recursive_filter(tau):
    """
    Une seule source de verite pour la dynamique de charge.

    Le modele bayesien ne doit pas redefinir la recurrence : il exprime la
    meme convolution, sinon le bug corrige d'un cote survit de l'autre.
    """
    rng = np.random.default_rng(0)
    trimp = rng.gamma(2, 25, 300)
    obs = np.array([40, 100, 180, 299])

    via_kernel = load_kernel(300, obs, tau) @ trimp
    via_filter = exponential_load(trimp, tau)[obs]
    assert np.allclose(via_kernel, via_filter, atol=1e-10)


def test_kernel_ignores_future_load():
    """Aucune charge posterieure a l'observation ne doit y contribuer."""
    trimp = np.zeros(100)
    trimp[80] = 500.0
    assert (load_kernel(100, np.array([50]), 42.0) @ trimp)[0] == 0.0


def test_kernel_weights_recent_load_more():
    recent, old = np.zeros(200), np.zeros(200)
    recent[190] = 100.0
    old[10] = 100.0
    kernel = load_kernel(200, np.array([199]), 42.0)
    assert (kernel @ recent)[0] > (kernel @ old)[0]


def test_a_shorter_time_constant_forgets_faster():
    trimp = np.zeros(200)
    trimp[20] = 100.0
    obs = np.array([199])
    assert (load_kernel(200, obs, 12.0) @ trimp)[0] < (load_kernel(200, obs, 42.0) @ trimp)[0]


# -- construction du modele ------------------------------------------------

def _simulated(n=400, tau1=45.0, tau2=10.0, k1=0.06, k2=0.03, p0=1.0,
               sigma=0.01, n_obs=40, seed=7):
    rng = np.random.default_rng(seed)
    trimp = rng.gamma(2, 25, n) * (rng.random(n) > 0.35)
    f = exponential_load(trimp, tau1)
    g = exponential_load(trimp, tau2)
    fz = (f - f.mean()) / f.std()
    gz = (g - g.mean()) / g.std()
    idx = np.sort(rng.choice(np.arange(60, n), n_obs, replace=False))
    y = p0 + k1 * fz[idx] - k2 * gz[idx] + rng.normal(0, sigma, n_obs)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    return (pd.DataFrame({"date": dates, "trimp": trimp}),
            pd.DataFrame({"date": dates[idx], "perf_index": y}),
            {'tau1': tau1, 'tau2': tau2, 'k1': k1, 'k2': k2, 'p0': p0})


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC absent")
def test_model_builds():
    daily, obs, _ = _simulated()
    model = build_banister_model(daily, obs)
    names = {v.name for v in model.free_RVs}
    # k1 n'est plus echantillonne directement : c'est k_net + k2.
    assert {"tau2", "delta", "k_net", "k2", "p0", "sigma_obs"} <= names
    assert "k1" not in names


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC absent")
def test_time_constants_cannot_be_negative():
    """
    Les priors d'origine etaient des lois normales non bornees. Une constante
    de temps negative rend e^{-1/tau} explosif et fait diverger
    l'echantillonnage.
    """
    daily, obs, _ = _simulated()
    model = build_banister_model(daily, obs)
    draws = pm_draw(model, ["tau1", "tau2"], 400)
    assert (draws["tau1"] > 0).all()
    assert (draws["tau2"] > 0).all()


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC absent")
def test_fitness_constant_always_exceeds_fatigue_constant():
    """
    Sans contrainte d'ordre les deux sont interchangeables dans la
    vraisemblance : les chaines permutent les etiquettes et r_hat ressort
    autour de 2.
    """
    daily, obs, _ = _simulated()
    draws = pm_draw(build_banister_model(daily, obs), ["tau1", "tau2"], 400)
    assert (draws["tau1"] > draws["tau2"]).all()


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC absent")
def test_fatigue_gain_is_not_forced_positive():
    """
    k2 doit pouvoir etre negatif. L'ajustement donne ce signe sur les donnees
    reelles, ce qui signale une endogeneite du moment de mesure : les seances
    rapides tombent pendant les blocs de charge. Contraindre le signe
    masquerait le resultat au lieu de le montrer.
    """
    daily, obs, _ = _simulated()
    draws = pm_draw(build_banister_model(daily, obs), ["k2"], 600)
    assert (draws["k2"] < 0).any()


def pm_draw(model, names, n):
    import pymc as pm

    with model:
        prior = pm.sample_prior_predictive(draws=n, random_seed=0)
    return {k: prior.prior[k].values.ravel() for k in names}


def test_too_few_observations_is_refused():
    daily, obs, _ = _simulated(n_obs=40)
    with pytest.raises(NotFittedError, match="cinq"):
        build_banister_model(daily, obs.head(2))


def test_observations_outside_the_calendar_are_dropped():
    daily, obs, _ = _simulated()
    stray = obs.copy()
    stray.loc[0, "date"] = pd.Timestamp("1999-01-01")
    if HAS_PYMC:
        assert len(build_banister_model(daily, stray).digital_twin_context["y"]) == len(obs) - 1


# -- le test qui prouve que l'inference fonctionne -------------------------

@pytest.mark.mcmc
@pytest.mark.slow
@pytest.mark.skipif(not HAS_PYMC, reason="PyMC absent")
def test_net_effect_is_better_determined_than_either_gain():
    """
    Le coeur du diagnostic d'identifiabilite.

    Fitness et fatigue etant correles, seul leur effet NET est contraint par
    les donnees. Sur l'historique reel, k1 et k2 ressortent correles a 0,979 :
    les donnees ne distinguent pas un gain de fitness eleve compense par une
    penalite de fatigue elevee d'une paire de gains modestes.
    """
    from src.layer_b_twin.bayesian_model import assess_identifiability, sample_model

    daily, obs, _ = _simulated()
    trace = sample_model(build_banister_model(daily, obs),
                         draws=600, tune=800, chains=2, target_accept=0.95)
    report = assess_identifiability(trace)

    assert report.summary.loc["k_net", "sd"] < report.summary.loc["k1", "sd"]
    assert report.summary.loc["k_net", "sd"] < report.summary.loc["k2", "sd"]
    assert abs(report.k1_k2_correlation) > 0.8


@pytest.mark.mcmc
@pytest.mark.slow
@pytest.mark.skipif(not HAS_PYMC, reason="PyMC absent")
def test_parameter_recovery_on_simulated_data():
    """
    Simuler depuis des parametres connus, puis verifier que le posterieur les
    retrouve. C'est la seule preuve qu'une implementation d'inference
    fonctionne, et la premiere chose qu'un lecteur exigera.
    """
    from src.layer_b_twin.bayesian_model import assess_identifiability, sample_model

    daily, obs, truth = _simulated()
    trace = sample_model(build_banister_model(daily, obs),
                         draws=600, tune=800, chains=2, target_accept=0.95)
    report = assess_identifiability(trace)

    assert report.max_r_hat < 1.05
    for name, true_value in truth.items():
        lo = report.summary.loc[name, "hdi_3%"]
        hi = report.summary.loc[name, "hdi_97%"]
        assert lo <= true_value <= hi, f"{name}: {true_value} hors de [{lo}, {hi}]"
