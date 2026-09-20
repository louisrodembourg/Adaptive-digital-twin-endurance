"""
Layer B — Digital Twin : Modèle d'État Bayésien (Banister étendu)

Implémente le modèle Fitness-Fatigue (Banister/Busso) dans un cadre
probabiliste PyMC pour quantifier l'incertitude sur l'état de l'athlète.

Architecture :
  - Fitness(t)     = Σᵢ TRIMP(i) × exp(-(t-i)/τ₁)
  - Fatigue(t)     = Σᵢ TRIMP(i) × exp(-(t-i)/τ₂)
  - Performance(t) = p₀ + k₁×Fitness(t) - k₂×Fatigue(t)
  - TSB(t)         = Fitness(t) - Fatigue(t)

Priors basés sur la littérature (Busso 2003, Clarke 2013) et calibrés
sur le record actuel de l'athlète (1h22 semi-marathon).

Outputs :
  - Courbes Fitness/Fatigue/TSB avec intervalles de confiance 95% (HDI)
  - ProbSuccess1h18 : probabilité de réussir l'objectif
  - Détection de sur-entraînement (TSB < -30) et sous-entraînement (TSB > 25)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# PyMC est importé conditionnellement (peut ne pas être installé)
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn(
        "PyMC non installé. Le modèle bayésien fonctionnera en mode déterministe.\n"
        "Installer avec : pip install pymc pytensor arviz",
        ImportWarning, stacklevel=2
    )


from ..config import load_params as _shared_load_params


def _load_params(config_path: str | Path | None = None) -> dict:
    """Parametres bruts — delegue au chargeur unique et mis en cache."""
    return _shared_load_params(config_path)


# ═══════════════════════════════════════════════════════════════
#  1. CALCUL DÉTERMINISTE FITNESS / FATIGUE
# ═══════════════════════════════════════════════════════════════

def compute_fitness_fatigue(
    trimp_series: np.ndarray,
    tau1: float = 42.0,
    tau2: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les courbes Fitness et Fatigue par convolution exponentielle.

    Fitness(t)  = Σᵢ₌₀ᵗ TRIMP(i) × exp(-(t-i)/τ₁)
    Fatigue(t)  = Σᵢ₌₀ᵗ TRIMP(i) × exp(-(t-i)/τ₂)

    Implémentation vectorisée via filtrage IIR équivalent :
      y[t] = TRIMP[t] + y[t-1] × exp(-1/τ)

    Parameters
    ----------
    trimp_series : array de TRIMP quotidien (une valeur par jour, 0 si repos)
    tau1 : constante de temps fitness (jours, défaut 42j)
    tau2 : constante de temps fatigue (jours, défaut 12j)

    Returns
    -------
    (fitness, fatigue) : deux arrays de même longueur que trimp_series
    """
    n = len(trimp_series)
    fitness = np.zeros(n)
    fatigue = np.zeros(n)

    decay1 = np.exp(-1.0 / tau1)
    decay2 = np.exp(-1.0 / tau2)

    for t in range(n):
        load = trimp_series[t]
        if t == 0:
            fitness[t] = load
            fatigue[t] = load
        else:
            fitness[t] = fitness[t-1] * decay1 + load
            fatigue[t] = fatigue[t-1] * decay2 + load

    return fitness, fatigue


def compute_performance(
    fitness: np.ndarray,
    fatigue: np.ndarray,
    p0: float = 0.0,
    k1: float = 1.0,
    k2: float = 2.0,
    nonlinear: bool = True,
) -> np.ndarray:
    """
    Calcule la performance estimée depuis les courbes Fitness/Fatigue.

    Performance(t) = p₀ + k₁×Fitness(t) - k₂×Fatigue(t)

    Avec non-linéarité optionnelle :
      - Si TSB < -30 → pénalité exponentielle (sur-entraînement)
      - Si TSB > 25  → plateau (rendements décroissants)

    Parameters
    ----------
    fitness, fatigue : arrays de même longueur
    p0 : performance baseline (valeur à charge nulle)
    k1 : gain fitness (influence positive)
    k2 : gain fatigue (influence négative, typiquement > k1)
    nonlinear : activer les corrections non-linéaires

    Returns
    -------
    np.ndarray : performance normalisée (1.0 = niveau record actuel)
    """
    tsb = fitness - fatigue
    perf = p0 + k1 * fitness - k2 * fatigue

    if nonlinear:
        # Pénalité sur-entraînement (TSB très négatif)
        overreach_mask = tsb < -30
        perf[overreach_mask] *= np.exp((tsb[overreach_mask] + 30) * 0.02)

        # Plateau rendements décroissants (TSB très positif = désentraînement)
        detrain_mask = tsb > 25
        perf[detrain_mask] *= (1 - (tsb[detrain_mask] - 25) * 0.005).clip(0.85)

    return perf


# ═══════════════════════════════════════════════════════════════
#  2. MODÈLE BAYÉSIEN PyMC
# ═══════════════════════════════════════════════════════════════

def build_banister_model(
    trimp_history: np.ndarray,
    perf_observations: list[tuple[int, float]],
    params: dict | None = None,
):
    """
    Construit le modèle probabiliste Banister dans PyMC.

    Parameters
    ----------
    trimp_history : array TRIMP quotidien (index 0 = jour le plus ancien)
    perf_observations : liste de (day_index, perf_normalized)
        perf_normalized : 1.0 = niveau record actuel (1h22), > 1.0 = amélioration
        ex: [(45, 1.0), (90, 1.02), (180, 1.04)]
    params : config (si None, charge params.yaml)

    Returns
    -------
    pm.Model si HAS_PYMC, sinon None

    Note
    ----
    Calibration de perf_normalized :
        perf_norm = semi_record_s / actual_time_s
        1h22 (4920s) → 4920/4920 = 1.0
        1h20 (4800s) → 4920/4800 = 1.025
        1h18 (4680s) → 4920/4680 = 1.051
    """
    if not HAS_PYMC:
        raise ImportError("PyMC requis. pip install pymc pytensor arviz")

    if params is None:
        params = _load_params()

    twin = params.get('twin', {})

    # Observations
    obs_days = np.array([d for d, _ in perf_observations])
    obs_perf = np.array([p for _, p in perf_observations])

    n_days = len(trimp_history)

    with pm.Model() as model:
        # ─── Priors (littérature + calibrés sur record 1h22) ───────
        tau1 = pm.Normal("tau1", mu=twin.get('tau1_prior_days', 42), sigma=5)
        tau2 = pm.Normal("tau2", mu=twin.get('tau2_prior_days', 12), sigma=2)
        k1   = pm.HalfNormal("k1", sigma=twin.get('k1_prior', 1.0))
        k2   = pm.HalfNormal("k2", sigma=twin.get('k2_prior', 2.0))
        p0   = pm.Normal("p0", mu=0.0, sigma=0.05)

        # ─── Calcul dynamique Fitness/Fatigue ──────────────────────
        # Implémentation via scan PyTensor pour compatibilité MCMC
        decay1 = pt.exp(-1.0 / tau1)
        decay2 = pt.exp(-1.0 / tau2)

        trimp_pt = pt.as_tensor_variable(trimp_history.astype(float))

        def step_fn(trimp_t, fit_prev, fat_prev, d1, d2):
            fit_t = fit_prev * d1 + trimp_t
            fat_t = fat_prev * d2 + trimp_t
            return fit_t, fat_t

        results, _ = pt.scan(
            fn=step_fn,
            sequences=[trimp_pt],
            outputs_info=[
                pt.zeros(()),  # fitness init
                pt.zeros(()),  # fatigue init
            ],
            non_sequences=[decay1, decay2],
        )

        fitness_curve = results[0]   # shape: (n_days,)
        fatigue_curve = results[1]

        # Extraire les valeurs aux jours d'observation
        fitness_obs = fitness_curve[obs_days]
        fatigue_obs = fatigue_curve[obs_days]

        perf_pred = p0 + k1 * fitness_obs - k2 * fatigue_obs

        # ─── Likelihood ────────────────────────────────────────────
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.03)
        pm.Normal("obs", mu=perf_pred, sigma=sigma_obs, observed=obs_perf)

    return model


def sample_model(
    model,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
):
    """
    Lance l'échantillonnage MCMC (NUTS) du modèle.

    Parameters
    ----------
    model : pm.Model retourné par build_banister_model
    draws : nombre d'échantillons post-tuning
    tune : nombre d'étapes de tuning (warm-up)
    chains : nombre de chaînes parallèles
    random_seed : graine pour reproductibilité

    Returns
    -------
    arviz.InferenceData
    """
    if not HAS_PYMC:
        raise ImportError("PyMC requis.")

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
        )
    return trace


# ═══════════════════════════════════════════════════════════════
#  3. EXTRACTION DES POSTERIORS ET COURBES
# ═══════════════════════════════════════════════════════════════

def extract_posterior_curves(
    trace,
    trimp_history: np.ndarray,
    params: dict | None = None,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """
    Reconstruit les courbes Fitness/Fatigue/TSB/Performance depuis
    les échantillons de la posterior.

    Returns
    -------
    DataFrame avec colonnes :
      day, fitness_mean, fitness_low, fitness_high,
      fatigue_mean, fatigue_low, fatigue_high,
      tsb_mean, tsb_low, tsb_high,
      perf_mean, perf_low, perf_high
    """
    if not HAS_PYMC:
        raise ImportError("PyMC requis.")

    if params is None:
        params = _load_params()

    # Extraire les paramètres de la posterior
    tau1_samples = trace.posterior["tau1"].values.flatten()
    tau2_samples = trace.posterior["tau2"].values.flatten()
    k1_samples   = trace.posterior["k1"].values.flatten()
    k2_samples   = trace.posterior["k2"].values.flatten()
    p0_samples   = trace.posterior["p0"].values.flatten()

    n_days = len(trimp_history)
    n_samples = len(tau1_samples)

    fitness_all = np.zeros((n_samples, n_days))
    fatigue_all = np.zeros((n_samples, n_days))
    perf_all    = np.zeros((n_samples, n_days))

    for i in range(n_samples):
        fit, fat = compute_fitness_fatigue(
            trimp_history, tau1_samples[i], tau2_samples[i]
        )
        perf = compute_performance(
            fit, fat, p0_samples[i], k1_samples[i], k2_samples[i]
        )
        fitness_all[i] = fit
        fatigue_all[i] = fat
        perf_all[i]    = perf

    tsb_all = fitness_all - fatigue_all

    def _hdi(arr, prob):
        lo = np.percentile(arr, (1 - prob) / 2 * 100, axis=0)
        hi = np.percentile(arr, (1 - (1 - prob) / 2) * 100, axis=0)
        return lo, hi

    fit_lo, fit_hi = _hdi(fitness_all, hdi_prob)
    fat_lo, fat_hi = _hdi(fatigue_all, hdi_prob)
    tsb_lo, tsb_hi = _hdi(tsb_all, hdi_prob)
    per_lo, per_hi = _hdi(perf_all, hdi_prob)

    df = pd.DataFrame({
        'day':          np.arange(n_days),
        'trimp':        trimp_history,

        'fitness_mean': fitness_all.mean(axis=0),
        'fitness_low':  fit_lo,
        'fitness_high': fit_hi,

        'fatigue_mean': fatigue_all.mean(axis=0),
        'fatigue_low':  fat_lo,
        'fatigue_high': fat_hi,

        'tsb_mean':     tsb_all.mean(axis=0),
        'tsb_low':      tsb_lo,
        'tsb_high':     tsb_hi,

        'perf_mean':    perf_all.mean(axis=0),
        'perf_low':     per_lo,
        'perf_high':    per_hi,
    })

    return df


def compute_prob_success(
    trace,
    trimp_history: np.ndarray,
    target_perf: float = 1.0513,  # 4920/4680 = 1h18
) -> dict:
    """
    Calcule la probabilité de succès 1h18 à J (dernier jour).

    target_perf = semi_record_s / target_time_s = 4920 / 4680 ≈ 1.051

    Returns
    -------
    dict {
        'prob_success': float [0-1],
        'perf_mean': float,
        'perf_hdi_low': float,
        'perf_hdi_high': float,
        'expected_time_s': float,
    }
    """
    if not HAS_PYMC:
        raise ImportError("PyMC requis.")

    tau1_s = trace.posterior["tau1"].values.flatten()
    tau2_s = trace.posterior["tau2"].values.flatten()
    k1_s   = trace.posterior["k1"].values.flatten()
    k2_s   = trace.posterior["k2"].values.flatten()
    p0_s   = trace.posterior["p0"].values.flatten()

    perf_final = []
    for i in range(len(tau1_s)):
        fit, fat = compute_fitness_fatigue(trimp_history, tau1_s[i], tau2_s[i])
        perf = compute_performance(fit, fat, p0_s[i], k1_s[i], k2_s[i])
        perf_final.append(perf[-1])

    perf_arr = np.array(perf_final)
    prob = float((perf_arr >= target_perf).mean())

    # Convertir perf moyenne en temps estimé
    params = _load_params()
    record_s = params['athlete']['semi_record_s']
    expected_time_s = record_s / perf_arr.mean() if perf_arr.mean() > 0 else np.nan

    return {
        'prob_success':   round(prob, 3),
        'perf_mean':      round(float(perf_arr.mean()), 4),
        'perf_hdi_low':   round(float(np.percentile(perf_arr, 2.5)), 4),
        'perf_hdi_high':  round(float(np.percentile(perf_arr, 97.5)), 4),
        'expected_time_s': round(float(expected_time_s), 0),
    }


# ═══════════════════════════════════════════════════════════════
#  4. MODE DÉTERMINISTE (fallback sans PyMC)
# ═══════════════════════════════════════════════════════════════

def run_deterministic(
    trimp_history: np.ndarray,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    Version déterministe du modèle Banister (sans incertitude).
    Utilisé quand PyMC n'est pas disponible ou pour les tests rapides.

    Returns
    -------
    DataFrame avec colonnes day, trimp, fitness, fatigue, tsb, perf_norm
    """
    if params is None:
        params = _load_params()

    twin = params.get('twin', {})
    tau1 = twin.get('tau1_prior_days', 42.0)
    tau2 = twin.get('tau2_prior_days', 12.0)
    k1   = twin.get('k1_prior', 1.0)
    k2   = twin.get('k2_prior', 2.0)

    fitness, fatigue = compute_fitness_fatigue(trimp_history, tau1, tau2)
    perf = compute_performance(fitness, fatigue, p0=0.0, k1=k1, k2=k2)
    tsb = fitness - fatigue

    return pd.DataFrame({
        'day':      np.arange(len(trimp_history)),
        'trimp':    trimp_history,
        'fitness':  np.round(fitness, 2),
        'fatigue':  np.round(fatigue, 2),
        'tsb':      np.round(tsb, 2),
        'perf_norm': np.round(perf, 4),
    })
